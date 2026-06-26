"""Train + evaluate SVA node circuits entirely in-framework (no nnsight, no MIB graph).

Phase 1: learn node scores via LlamaAttributionHooks + build_mask + learn_scores on SVADataset
         (clean/patch minimal pairs, logit-diff objective) — same path as eval_mib.py.
Phase 2: faithfulness sparsity-sweep via the same hooker (keep top-k CLEAN, ablate the rest to
         the patch counterfactual; normalized recovery of the clean logit-diff).

Node sets (--nodes): "mlp" = per-(layer,pos,neuron) MLP acts; "mlp+attn_dim" = that plus
per-(layer,pos,dim) attention pre-out (o_proj input) — i.e. mlp acts per-neuron and attn
pre-out per-dim.
"""
import argparse, json, logging, random, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import learn_scores, sparsity_sweep
from learning_to_attribute.data import SVADataset
from learning_to_attribute.models import LlamaAttributionHooks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_FULLNAMES = {"gpt2": "gpt2", "qwen2.5": "Qwen/Qwen2.5-0.5B",
                   "gemma2": "google/gemma-2-2b", "llama3": "meta-llama/Llama-3.1-8B"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3", choices=list(MODEL_FULLNAMES))
    p.add_argument("--task", required=True)            # nounpp | rc | simple | within_rc
    p.add_argument("--nodes", default="mlp", choices=["mlp", "mlp+attn_dim"])
    p.add_argument("--variant", default="hard_topk",
                   choices=["topk", "hard_topk", "hard_topk_identity"])  # build_mask gate
    p.add_argument("--mode", default="sufficient", choices=["sufficient", "necessary"])
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--k-schedule", default="log", choices=["uniform", "log"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n_iters", type=int, default=30)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-examples", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/sva")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); torch.manual_seed(args.seed)

    name = MODEL_FULLNAMES[args.model]
    logger.info("Loading %s ...", name)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    for pp in hf.parameters():
        pp.requires_grad_(False)

    train = SVADataset(args.task, tok, split="train")
    test = SVADataset(args.task, tok, split="test")

    # seq_len for the per-position node layout: use the modal clean-prompt token length; the
    # train/eval loops only sample pairs of exactly this length (so the mask indices line up).
    lens = Counter(tok(train[i][0], return_tensors="pt").input_ids.shape[1] for i in range(min(400, len(train))))
    seq_len = lens.most_common(1)[0][0]
    logger.info("seq_len=%d (modal clean length; %s)", seq_len, dict(lens))

    corrupt_topk = args.mode == "necessary"
    hooker = LlamaAttributionHooks(hf, args.nodes, seq_len=seq_len,
                                   sufficient=corrupt_topk, include_input=False)
    total = hooker.total
    logger.info("Nodes (%s): %s", args.nodes, hooker.describe())
    hooker.register_hooks()

    def sample_batch(ds, B, n):
        cl, co, ci, ii = [], [], [], []
        tries = 0
        while len(cl) < B and tries < B * 20:
            tries += 1
            clean, corr, lab = ds[random.randint(0, n - 1)]
            a = tok(clean, return_tensors="pt").input_ids
            b = tok(corr, return_tensors="pt").input_ids
            if a.shape[1] != seq_len or b.shape[1] != seq_len:
                continue
            cl.append(clean); co.append(corr); ci.append(lab[0]); ii.append(lab[1])
        return cl, co, ci, ii

    def forward_logit_diff(cleans, corrupteds, ci, ii, mask, sufficient):
        bt = tok(cleans, return_tensors="pt", padding=True).to(device)
        st = tok(corrupteds, return_tensors="pt", padding=True).to(device)
        last = bt.attention_mask.sum(1) - 1
        hooker.cache_cf_activations(st.input_ids)
        old_suf = hooker.sufficient; hooker.sufficient = sufficient
        hooker.mask = mask
        logits = hf(bt.input_ids, attention_mask=bt.attention_mask).logits.float()
        hooker.sufficient = old_suf
        B = len(cleans)
        ll = logits[torch.arange(B, device=device), last]
        cor = torch.tensor(ci, device=device); inc = torch.tensor(ii, device=device)
        return ll[torch.arange(B, device=device), cor] - ll[torch.arange(B, device=device), inc]

    n_train = len(train)
    def loss_fn(mask):
        cl, co, ci, ii = sample_batch(train, args.train_batch_size, n_train)
        if not cl:
            return None
        d = forward_logit_diff(cl, co, ci, ii, mask, sufficient=corrupt_topk)
        return d.mean() if corrupt_topk else -d.mean()

    logger.info("Training %d steps (%s gate, %s, %s, k=%s)...", args.steps, args.variant,
                args.mode, args.optimizer, args.k_schedule)
    res = learn_scores(total, loss_fn, steps=args.steps, variant=args.variant,
                       k_schedule=args.k_schedule, T=args.T, n_iters=args.n_iters,
                       lr=args.lr, optimizer=args.optimizer, use_bias=False, device=device)
    scores = res.scores.detach()

    # ---- Phase 2: sparsity sweep, BOTH directions, counterfactual (patch) ablation ----
    #   iso  (sufficiency): keep top-k CLEAN, corrupt the complement  -> recovery curve
    #   cause(necessity):   corrupt top-k, keep the complement clean  -> breakage curve
    # Same top-k ranking; only the hooker `sufficient` flag flips. Complement/top-k are ablated
    # to each example's own counterfactual (patch), matching training (mean-abl deferred).
    ec, eco, eci, eii = [], [], [], []
    for i in range(len(test)):
        clean, corr, lab = test[i]
        if tok(clean, return_tensors="pt").input_ids.shape[1] != seq_len: continue
        if tok(corr, return_tensors="pt").input_ids.shape[1] != seq_len: continue
        ec.append(clean); eco.append(corr); eci.append(lab[0]); eii.append(lab[1])
        if len(ec) >= args.eval_examples: break
    logger.info("Eval on %d test pairs (len=%d)", len(ec), seq_len)

    @torch.no_grad()
    def eval_ld(mask, sufficient):
        out = []
        for s in range(0, len(ec), 20):
            d = forward_logit_diff(ec[s:s+20], eco[s:s+20], eci[s:s+20], eii[s:s+20],
                                   mask.to(device), sufficient=sufficient)
            out.append(d)
        return torch.cat(out).mean().item()

    # FM = all clean, F0 = all patched (direction-independent; verify via both conventions)
    FM = eval_ld(torch.ones(total), sufficient=False)
    F0 = eval_ld(torch.zeros(total), sufficient=False)
    denom = (FM - F0) or 1e-9
    logger.info("F(clean)=%.3f  F(patch)=%.3f", FM, F0)

    sparsities = sorted(set(float(10 ** x) for x in np.linspace(np.log10(1.0/total), 0.0, 24)))

    def auc_of(ys, xs):
        lx = np.log10(xs); ya = np.asarray(ys, float)
        return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))

    # iso/faithfulness: top-k clean (sufficient=False), normalized recovery
    iso = sparsity_sweep(scores, total, sparsities,
                         lambda hm: {"v": (eval_ld(hm, False) - F0) / denom},
                         device=device, include_random=False)
    # cause/completeness: corrupt top-k (sufficient=True); 1 - normalized-remaining = breakage
    cause = sparsity_sweep(scores, total, sparsities,
                           lambda hm: {"v": (eval_ld(hm, True) - F0) / denom},
                           device=device, include_random=False)
    xs = [s * total for s in sparsities]
    faith = iso["learned"]["v"]; compl = cause["learned"]["v"]
    faith_auc = auc_of(faith, xs); cause_auc = auc_of(compl, xs)
    hooker.remove_hooks()

    out = dict(task=args.task, model=args.model, nodes=args.nodes, variant=args.variant,
               mode=args.mode, optimizer=args.optimizer, k_schedule=args.k_schedule,
               total=total, seq_len=seq_len, F_clean=FM, F_patch=F0, n_nodes=xs,
               faith_auc=faith_auc, faith_max=max(faith), faithfulness=faith,
               cause_auc=cause_auc, cause_curve=compl)
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"{args.task}_{args.model}_{args.nodes.replace('+','-')}_{args.mode}_{args.variant}_{args.optimizer}.json"
    json.dump(out, open(fn, "w"), indent=2)
    logger.info("iso/faith AUC=%.3f (fmax %.3f) | cause AUC=%.3f | total=%d -> %s",
                faith_auc, max(faith), cause_auc, total, fn)


if __name__ == "__main__":
    main()

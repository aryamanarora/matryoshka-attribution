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


def gradient_scores(hf, hooker, ds, seq_len, total, tok, device, n_examples=100, relp=False, ig_steps=1):
    """Closed-form gradient attribution (IxG = grad x delta) over the hooker's node layout.

    Captures the clean activation at each node module (down_proj / o_proj input) with a
    forward-pre-hook (retain_grad), runs a clean forward + logit-diff backward, and scores each
    node by g . (clean - patch), summed over a batch. relp=True applies the RelP modified
    backward first (LN-freeze + MLP gate rule + QK-detach). [RelP backward not yet ported.]
    """
    if relp:
        from learning_to_attribute.grad_attribution import install_relp, revert_relp
        install_relp(hf)
    layers = hf.model.layers
    use_attn = hooker.mask_type == "mlp+attn_dim"
    N, H, P = hooker.intermediate_size, hooker.hidden_size, seq_len

    # collect a same-length batch of clean/patch pairs
    cl, co, ci, ii = [], [], [], []
    i = 0
    while len(cl) < n_examples and i < len(ds):
        clean, corr, lab = ds[i]; i += 1
        if tok(clean, return_tensors="pt").input_ids.shape[1] != seq_len: continue
        if tok(corr, return_tensors="pt").input_ids.shape[1] != seq_len: continue
        cl.append(clean); co.append(corr); ci.append(lab[0]); ii.append(lab[1])

    bt = tok(cl, return_tensors="pt", padding=True).to(device)
    bid, bam = bt.input_ids, bt.attention_mask
    last = bam.sum(1) - 1
    B = len(cl)
    cor = torch.tensor(ci, device=device); inc = torch.tensor(ii, device=device)

    def capture(ids, am, want_grad, embed_override=None):
        store = {}; handles = []
        def mk(li, kind):
            def hook(mod, args):
                x = args[0]
                if want_grad:
                    x.requires_grad_(True); x.retain_grad()
                store[(li, kind)] = x
                return (x,) + tuple(args[1:])
            return hook
        for li in range(len(layers)):
            handles.append(layers[li].mlp.down_proj.register_forward_pre_hook(mk(li, "mlp")))
            if use_attn:
                handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li, "attn")))
        if embed_override is not None:
            handles.append(hf.model.embed_tokens.register_forward_hook(
                lambda mod, inp, out: embed_override))
        logits = hf(ids, attention_mask=am).logits.float()
        for h in handles: h.remove()
        return store, logits

    def metric_of(logits):
        ll = logits[torch.arange(B, device=device), last]
        return (ll[torch.arange(B, device=device), cor] - ll[torch.arange(B, device=device), inc]).sum()

    # cached clean & patch node acts (no grad) -> delta
    pt = tok(co, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        clean_acts, _ = capture(bid, bam, False)
        patch_acts, _ = capture(pt.input_ids, pt.attention_mask, False)
    clean_acts = {k: v.detach() for k, v in clean_acts.items()}
    patch_acts = {k: v.detach() for k, v in patch_acts.items()}

    # embeddings for the IG path (interpolate clean->patch input embedding, downstream live)
    emb_override = None
    if ig_steps > 1:
        cap = {}
        h = hf.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("e", o.detach()))
        with torch.no_grad(): hf(bid, attention_mask=bam); ec = cap["e"]
        with torch.no_grad(): hf(pt.input_ids, attention_mask=pt.attention_mask); ep = cap["e"]
        h.remove()

    grad_acc = {k: torch.zeros_like(v) for k, v in clean_acts.items()}
    alphas = [s / ig_steps for s in range(ig_steps)] if ig_steps > 1 else [0.0]
    for alpha in alphas:
        if ig_steps > 1:
            emb_override = (1 - alpha) * ec + alpha * ep
        store_g, logits = capture(bid, bam, True, embed_override=emb_override)
        metric_of(logits).backward()
        for k in grad_acc:
            grad_acc[k] += store_g[k].grad
    for k in grad_acc:
        grad_acc[k] /= len(alphas)

    scores = torch.zeros(total)
    for li in range(len(layers)):
        eff = (grad_acc[(li, "mlp")] * (clean_acts[(li, "mlp")] - patch_acts[(li, "mlp")])).sum(0)
        off = li * P * N; scores[off:off + P * N] = eff.reshape(-1).cpu()
        if use_attn:
            eff = (grad_acc[(li, "attn")] * (clean_acts[(li, "attn")] - patch_acts[(li, "attn")])).sum(0)
            off = hooker.mlp_total + li * P * H; scores[off:off + P * H] = eff.reshape(-1).cpu()
    if relp:
        revert_relp(hf)
    return scores.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3", choices=list(MODEL_FULLNAMES))
    p.add_argument("--task", required=True)            # nounpp | rc | simple | within_rc
    p.add_argument("--method", default="mattr", choices=["mattr", "ixg", "relp", "ig"])
    p.add_argument("--ig-steps", type=int, default=10, help="IG integration steps (input-embedding path)")
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

    if args.method in ("ixg", "relp", "ig"):
        scores = gradient_scores(hf, hooker, train, seq_len, total, tok, device,
                                 n_examples=args.eval_examples, relp=(args.method == "relp"),
                                 ig_steps=args.ig_steps if args.method == "ig" else 1)
    else:
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
    def eval_metrics(mask, sufficient):
        # base = clean/correct answer (ci); source = patch answer (ii)
        LB, LS, PB, PS = [], [], [], []
        for s in range(0, len(ec), 20):
            bt = tok(ec[s:s+20], return_tensors="pt", padding=True).to(device)
            st = tok(eco[s:s+20], return_tensors="pt", padding=True).to(device)
            last = bt.attention_mask.sum(1) - 1
            hooker.cache_cf_activations(st.input_ids)
            old = hooker.sufficient; hooker.sufficient = sufficient; hooker.mask = mask.to(device)
            logits = hf(bt.input_ids, attention_mask=bt.attention_mask).logits.float()
            hooker.sufficient = old
            B = bt.input_ids.shape[0]; ar = torch.arange(B, device=device)
            ll = logits[ar, last]; probs = ll.softmax(-1)
            cor = torch.tensor(eci[s:s+20], device=device); inc = torch.tensor(eii[s:s+20], device=device)
            LB.append(ll[ar, cor]); LS.append(ll[ar, inc]); PB.append(probs[ar, cor]); PS.append(probs[ar, inc])
        lb = torch.cat(LB); ls = torch.cat(LS); pb = torch.cat(PB); ps = torch.cat(PS)
        ld = lb - ls
        return {"logit_diff": ld.mean().item(),
                "p_base": pb.mean().item(), "p_source": ps.mean().item(),
                "acc_base": (lb > ls).float().mean().item(),       # 1[p(base) > p(source)]
                "acc_source": (ls > lb).float().mean().item(),     # 1[p(source) > p(base)]
                "log_odds_ratio": ld.mean().item(),                # mean log(p_base/p_source) = logit-diff
                "odds_ratio": float(torch.exp(ld.mean()))}         # geometric-mean odds (stable)

    FM = eval_metrics(torch.ones(total), sufficient=False)["logit_diff"]   # all clean
    F0 = eval_metrics(torch.zeros(total), sufficient=False)["logit_diff"]  # all patched
    denom = (FM - F0) or 1e-9
    logger.info("F(clean)=%.3f  F(patch)=%.3f", FM, F0)

    sparsities = sorted(set(float(10 ** x) for x in np.linspace(np.log10(1.0/total), 0.0, 24)))

    def auc_of(ys, xs):
        lx = np.log10(xs); ya = np.asarray(ys, float)
        return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))

    def metrics_at(mask, sufficient):
        m = eval_metrics(mask, sufficient)
        m["faithfulness"] = (m["logit_diff"] - F0) / denom   # normalized logit-diff
        return m

    # iso (sufficiency): keep top-k clean ; cause (necessity): corrupt top-k
    iso = sparsity_sweep(scores, total, sparsities, lambda hm: metrics_at(hm, False),
                         device=device, include_random=False)["learned"]
    cause = sparsity_sweep(scores, total, sparsities, lambda hm: metrics_at(hm, True),
                           device=device, include_random=False)["learned"]
    xs = [s * total for s in sparsities]
    faith = iso["faithfulness"]; compl = cause["faithfulness"]
    faith_auc = auc_of(faith, xs); cause_auc = auc_of(compl, xs)
    hooker.remove_hooks()

    out = dict(task=args.task, model=args.model, nodes=args.nodes, variant=args.variant,
               mode=args.mode, optimizer=args.optimizer, k_schedule=args.k_schedule,
               total=total, seq_len=seq_len, F_clean=FM, F_patch=F0, n_nodes=xs,
               faith_auc=faith_auc, faith_max=max(faith), faithfulness=faith,
               cause_auc=cause_auc, cause_curve=compl,
               iso_metrics=iso, cause_metrics=cause)   # full per-metric curves, both directions
    out["intermediate_size"] = hooker.intermediate_size
    out["hidden_size"] = hooker.hidden_size
    out["num_layers"] = hooker.num_layers
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    tag = args.method if args.method != "mattr" else f"{args.mode}_{args.variant}_{args.optimizer}"
    fn = outdir / f"{args.task}_{args.model}_{args.nodes.replace('+','-')}_{tag}.json"
    torch.save(scores.cpu(), fn.with_suffix(".scores.pt"))
    json.dump(out, open(fn, "w"), indent=2)
    logger.info("iso/faith AUC=%.3f (fmax %.3f) | cause AUC=%.3f | total=%d -> %s",
                faith_auc, max(faith), cause_auc, total, fn)


if __name__ == "__main__":
    main()

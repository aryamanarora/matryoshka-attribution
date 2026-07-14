"""Train + evaluate SVA node circuits entirely in-framework (no nnsight, no MIB graph).

Phase 1: learn node scores via LlamaAttributionHooks + build_mask + learn_scores on SVADataset
         (clean/patch minimal pairs, logit-diff objective) — same path as eval_mib.py.
Phase 2: faithfulness sparsity-sweep via the same hooker (keep top-k CLEAN, ablate the rest to
         the patch counterfactual; normalized recovery of the clean logit-diff).

Node sets (--nodes): "mlp" = per-(layer,pos,neuron) MLP acts; "mlp+attn_dim" = that plus
per-(layer,pos,dim) attention pre-out (o_proj input) — i.e. mlp acts per-neuron and attn
pre-out per-dim.
"""
import argparse, json, logging, math, random, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import learn_scores, sparsity_sweep
from learning_to_attribute.schedules import AdaptiveLogK, FixedK
from learning_to_attribute.losses import attribution_loss, resolve_direction, LOSS_CHOICES
from learning_to_attribute.data import SVADataset, CausalGymDataset
from learning_to_attribute.models import LlamaAttributionHooks


# span-last token positions per string, keyed by the (cleaned) string. Lets the span-mode
# forwards recover each example's per-span node positions without threading them everywhere.
SPAN_LAST = {}      # cleaned string -> [last-tok-pos per content span]
NUM_SPANS = None    # constant content-span count for the active task


def _content_spans(spans):
    """Drop the gpt2 <|endoftext|> prefix span; lstrip the first remaining span (matches the
    cleaned string used downstream)."""
    cs = [s for s in spans if s != "<|endoftext|>"]
    if cs:
        cs = [cs[0].lstrip()] + cs[1:]
    return cs


def _span_last(tokenizer, content_spans):
    """Last token position of each content span in tokenizer(join(content_spans))."""
    text = "".join(content_spans)
    n_with_special = tokenizer(text, return_tensors="pt").input_ids.shape[1]
    bos = n_with_special - len(tokenizer.tokenize(text))   # leading special-token offset
    pos, last = bos, []
    for s in content_spans:
        pos += len(tokenizer.tokenize(s))
        last.append(pos - 1)
    return last


class CGDataset:
    """CausalGym task as a fixed list of (clean, corrupted, [base_id, source_id]) pairs.
    Drop-in for SVADataset; strips the gpt2 <|endoftext|> prefix (the model tokenizer adds BOS).
    Also records per-span last-token positions (in SPAN_LAST) for span-tied attribution."""
    def __init__(self, task, tokenizer, n=2000, seed=42):
        global NUM_SPANS
        cg = CausalGymDataset(f"syntaxgym/{task}", seed=seed)
        self.recs = []
        for _ in range(n):
            p = cg.sample_pair()
            bcs, scs = _content_spans(p.base_spans), _content_spans(p.src_spans)
            clean, corr = "".join(bcs), "".join(scs)
            if NUM_SPANS is None:
                NUM_SPANS = len(bcs)
            if clean not in SPAN_LAST:
                SPAN_LAST[clean] = _span_last(tokenizer, bcs)
            if corr not in SPAN_LAST:
                SPAN_LAST[corr] = _span_last(tokenizer, scs)
            bid = tokenizer(p.base_label).input_ids[-1]
            sid = tokenizer(p.src_label).input_ids[-1]
            self.recs.append((clean, corr, [bid, sid]))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return self.recs[i]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_FULLNAMES = {"gpt2": "gpt2", "qwen2.5": "Qwen/Qwen2.5-0.5B",
                   "gemma2": "google/gemma-2-2b", "llama3": "meta-llama/Llama-3.1-8B"}


def gradient_scores(hf, hooker, ds, seq_len, total, tok, device, n_examples=100, relp=False, ig_steps=1,
                    loss="logit_diff", hinge_margin=2.0, acc_temp=1.0, conductance=False):
    """Closed-form gradient attribution (IxG = grad x delta) over the hooker's node layout.

    Captures the clean activation at each node module (down_proj / o_proj input) with a
    forward-pre-hook (retain_grad), runs a clean forward + logit-diff backward, and scores each
    node by g . (clean - patch), summed over a batch. relp=True applies the RelP modified
    backward first (LN-freeze + MLP gate rule + QK-detach). [RelP backward not yet ported.]
    """
    if hooker.mask_type in ("mlp_sae_span", "resid_sae_span", "das_mlp_span", "das_resid_span"):
        raise NotImplementedError("gradient attribution not supported for SAE/DAS nodes; use --method mattr")
    if relp:
        from learning_to_attribute.grad_attribution import install_relp, revert_relp
        install_relp(hf)
    layers = hf.model.layers
    use_attn = hooker.mask_type in ("mlp+attn_dim", "mlp+attn_head", "node", "mlp+attn_span", "mlp+attn_head_span")
    head_nonspan = hooker.mask_type == "mlp+attn_head"   # per-(pos, head), fixed-length
    is_node = hooker.mask_type == "node"                 # MIB granularity: mlp block + attn head
    span = hooker.mask_type in ("mlp_span", "mlp+attn_span", "mlp+attn_head_span")
    span_attn = hooker.mask_type == "mlp+attn_span"
    span_head = hooker.mask_type == "mlp+attn_head_span"
    N, H, P = hooker.intermediate_size, hooker.hidden_size, seq_len

    # collect a batch of clean/patch pairs (span mode: variable length; else fixed seq_len)
    cl, co, ci, ii = [], [], [], []
    i = 0
    while len(cl) < n_examples and i < len(ds):
        clean, corr, lab = ds[i]; i += 1
        if not span and not is_node:   # node is position-agnostic -> variable length OK
            if tok(clean, return_tensors="pt").input_ids.shape[1] != seq_len: continue
            if tok(corr, return_tensors="pt").input_ids.shape[1] != seq_len: continue
        elif is_node:   # node: clean/corrupted must match length (interpolation alignment)
            if tok(clean, return_tensors="pt").input_ids.shape[1] != \
               tok(corr, return_tensors="pt").input_ids.shape[1]: continue
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
        # gradient-attribution target = goodness (= -loss), scored by g . (clean - patch).
        # sufficiency direction (corrupt_topk=False): reward recovering the BASE answer. For
        # loss=logit_diff this is exactly (logit_base - logit_source), matching the prior default.
        ll = logits[torch.arange(B, device=device), last]
        return -attribution_loss(loss, ll, cor, inc, corrupt_topk=False,
                                 hinge_margin=hinge_margin, acc_temp=acc_temp)

    # cached clean & patch node acts (no grad) -> delta
    pt = tok(co, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        clean_acts, _ = capture(bid, bam, False)
        patch_acts, _ = capture(pt.input_ids, pt.attention_mask, False)
    clean_acts = {k: v.detach() for k, v in clean_acts.items()}
    patch_acts = {k: v.detach() for k, v in patch_acts.items()}

    # embeddings for the IG path (interpolate clean->patch input embedding, downstream live)
    emb_override = None
    if ig_steps > 1 or conductance or hooker.include_input:
        cap = {}
        h = hf.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("e", o.detach()))
        with torch.no_grad(): hf(bid, attention_mask=bam); ec = cap["e"]
        with torch.no_grad(): hf(pt.input_ids, attention_mask=pt.attention_mask); ep = cap["e"]
        h.remove()

    def input_node_effect():
        # score for the input-embedding node (index 0 when include_input): grad(emb).(clean-patch),
        # averaged over the IG path. The input embedding is interpolated LINEARLY, so its IG and
        # conductance coincide; ixg (ig_steps=1) is grad at the clean embedding.
        S = ig_steps if ig_steps > 1 else 1
        g_acc = torch.zeros_like(ec)
        for step in range(1, S + 1):
            eo = (ep + (step / S) * (ec - ep)).detach().requires_grad_(True)   # step=S -> clean
            hh = hf.model.embed_tokens.register_forward_hook(lambda m, i, o: eo)
            metric_of(hf(bid, attention_mask=bam).logits.float()).backward()
            hh.remove()
            g_acc += eo.grad
        return float((g_acc / S * (ec - ep)).sum())

    if conductance:
        # CONDUCTANCE (local-delta): proper Riemann sum along the ACTUAL (nonlinear) activation
        # trajectory as the input embedding goes clean->patch. Instead of pulling the endpoint
        # delta (clean-patch) out of the integral (that is exact only for alpha-linear nodes),
        # accumulate the realized per-step increment  sum_k grad(a_k) . (a(a_{k-1}) - a(a_k)).
        # fp32 accumulation. Node granularity only (span/per-pos cross-alignment not handled).
        assert is_node, "conductance implemented for --nodes node only"
        S = ig_steps if ig_steps > 1 else 30
        prev = {k: clean_acts[k].float() for k in clean_acts}         # a(alpha_0) = clean
        cond = {k: torch.zeros(v.shape, device=device, dtype=torch.float32) for k, v in clean_acts.items()}
        for step in range(1, S + 1):
            emb_override = (1 - step / S) * ec + (step / S) * ep       # clean -> patch
            store_g, logits = capture(bid, bam, True, embed_override=emb_override)
            metric_of(logits).backward()
            for k in cond:
                cur = store_g[k].detach().float()
                cond[k] += store_g[k].grad.float() * (prev[k] - cur)  # grad(a_k) . (a_{k-1}-a_k)
                prev[k] = cur
        off0, nh, Hd, L = hooker._node_offset, hooker.num_heads, hooker.head_dim, len(layers)
        scores = torch.zeros(total)
        for li in range(L):
            scores[off0 + L * nh + li] = cond[(li, "mlp")].sum().cpu()
            c = cond[(li, "attn")]; Bn, Pn = c.shape[0], c.shape[1]
            scores[off0 + li * nh:off0 + (li + 1) * nh] = c.view(Bn, Pn, nh, Hd).sum(-1).sum((0, 1)).cpu()
        if hooker.include_input:
            scores[0] = input_node_effect()
        if relp:
            revert_relp(hf)
        return scores.to(device)

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

    tied = hooker.mask_type == "mlp_tied"
    scores = torch.zeros(total)
    if is_node:
        # MIB node granularity: one scalar per MLP block per layer (g.delta summed over the
        # whole intermediate block + positions; == masking the MLP output, down_proj linear),
        # and one per attention head per layer (g.delta summed over head_dim + positions).
        # Layout mirrors the hooker: [offset][attn: L*nh heads][mlp: L blocks].
        off0 = hooker._node_offset
        nh, Hd, L = hooker.num_heads, hooker.head_dim, len(layers)
        for li in range(L):
            cm = grad_acc[(li, "mlp")] * (clean_acts[(li, "mlp")] - patch_acts[(li, "mlp")])
            scores[off0 + L * nh + li] = cm.sum().cpu()                # [B,P,N] -> scalar
            ga, ca, pa = grad_acc[(li, "attn")], clean_acts[(li, "attn")], patch_acts[(li, "attn")]
            Bn, Pn = ga.shape[0], ga.shape[1]
            effh = (ga.view(Bn, Pn, nh, Hd)
                    * (ca.view(Bn, Pn, nh, Hd) - pa.view(Bn, Pn, nh, Hd))).sum(-1).sum((0, 1))
            scores[off0 + li * nh:off0 + (li + 1) * nh] = effh.cpu()   # [nh]
        if hooker.include_input:
            scores[0] = input_node_effect()
        if relp:
            revert_relp(hf)
        return scores.to(device)
    if span:
        # per-(layer, span, neuron): node at each span's last token, cross-aligned base<-src.
        # effect = grad[base_last] . (clean[base_last] - patch[src_last]).
        S = hooker.num_spans
        bi = torch.tensor([SPAN_LAST[c] for c in cl], device=device)[:, :, None].expand(-1, -1, N)
        si = torch.tensor([SPAN_LAST[c] for c in co], device=device)[:, :, None].expand(-1, -1, N)
        bih = torch.tensor([SPAN_LAST[c] for c in cl], device=device)[:, :, None].expand(-1, -1, H)
        sih = torch.tensor([SPAN_LAST[c] for c in co], device=device)[:, :, None].expand(-1, -1, H)
        if span_head:
            nh, Hd = hooker.num_heads, hooker.head_dim
            bl = torch.tensor([SPAN_LAST[c] for c in cl], device=device)  # [B,S]
            sl = torch.tensor([SPAN_LAST[c] for c in co], device=device)
            bih4 = bl[:, :, None, None].expand(-1, -1, nh, Hd)
            sih4 = sl[:, :, None, None].expand(-1, -1, nh, Hd)
        for li in range(len(layers)):
            g = grad_acc[(li, "mlp")].gather(1, bi)
            c = clean_acts[(li, "mlp")].gather(1, bi)
            p = patch_acts[(li, "mlp")].gather(1, si)
            eff = (g * (c - p)).sum(0)             # [S, N]
            off = li * S * N; scores[off:off + S * N] = eff.reshape(-1).cpu()
            if span_attn:
                ga = grad_acc[(li, "attn")].gather(1, bih)
                ca = clean_acts[(li, "attn")].gather(1, bih)
                pa = patch_acts[(li, "attn")].gather(1, sih)
                effa = (ga * (ca - pa)).sum(0)     # [S, H]
                offa = hooker.mlp_span_total + li * S * H
                scores[offa:offa + S * H] = effa.reshape(-1).cpu()
            if span_head:
                nh, Hd = hooker.num_heads, hooker.head_dim
                Bn = grad_acc[(li, "attn")].shape[0]
                g4 = grad_acc[(li, "attn")].view(Bn, -1, nh, Hd).gather(1, bih4)
                c4 = clean_acts[(li, "attn")].view(Bn, -1, nh, Hd).gather(1, bih4)
                p4 = patch_acts[(li, "attn")].view(Bn, -1, nh, Hd).gather(1, sih4)
                effh = (g4 * (c4 - p4)).sum(-1).sum(0)   # sum head_dim, then batch -> [S, nh]
                offh = hooker.mlp_span_total + li * S * nh
                scores[offh:offh + S * nh] = effh.reshape(-1).cpu()
        if relp:
            revert_relp(hf)
        return scores.to(device)
    for li in range(len(layers)):
        contrib = grad_acc[(li, "mlp")] * (clean_acts[(li, "mlp")] - patch_acts[(li, "mlp")])  # [B,P,N]
        if tied:
            # tie across token positions (MIB node convention): sum the g.delta attribution
            # over both batch and positions -> one score per (layer, neuron).
            eff = contrib.sum(0).sum(0)            # [N]
            off = li * N; scores[off:off + N] = eff.cpu()
            continue
        eff = contrib.sum(0)
        off = li * P * N; scores[off:off + P * N] = eff.reshape(-1).cpu()
        if use_attn:
            ga, ca, pa = grad_acc[(li, "attn")], clean_acts[(li, "attn")], patch_acts[(li, "attn")]
            if head_nonspan:
                # per-(pos, head): reshape o_proj input to heads, sum g.delta over head_dim.
                nh, Hd = hooker.num_heads, hooker.head_dim
                Bn = ga.shape[0]
                effh = (ga.view(Bn, P, nh, Hd)
                        * (ca.view(Bn, P, nh, Hd) - pa.view(Bn, P, nh, Hd))).sum(-1).sum(0)  # [P, nh]
                off = hooker.mlp_total + li * P * nh
                scores[off:off + P * nh] = effh.reshape(-1).cpu()
            else:  # mlp+attn_dim: per-(pos, dim)
                eff = (ga * (ca - pa)).sum(0)
                off = hooker.mlp_total + li * P * H; scores[off:off + P * H] = eff.reshape(-1).cpu()
    if relp:
        revert_relp(hf)
    return scores.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3", choices=list(MODEL_FULLNAMES))
    p.add_argument("--task", required=True)            # sva: nounpp|rc|simple|within_rc ; causalgym: e.g. npi_any_subj-relc
    p.add_argument("--dataset", default="sva", choices=["sva", "causalgym", "mib"])
    p.add_argument("--method", default="mattr", choices=["mattr", "ixg", "relp", "ig", "conductance", "random"])
    p.add_argument("--ig-steps", type=int, default=10, help="IG integration steps (input-embedding path)")
    p.add_argument("--train-eval-every", type=int, default=0,
                   help="MAttr: every N steps, run the FULL eval-metric suite on a fixed tiny "
                        "train subset and log it (unconfounded by the per-step k). 0 = off.")
    p.add_argument("--train-eval-examples", type=int, default=20,
                   help="# fixed train examples for the --train-eval-every probe.")
    p.add_argument("--include-input", action="store_true",
                   help="score + ablate the input-embedding node (node substrate only), matching "
                        "MIB's graph which includes an input node. Adds 1 node at index 0.")
    p.add_argument("--mattr-ig-steps", type=int, default=1,
                   help="MAttr-IG: integrate dL/dmask over this many baseline(CF)->clean mask "
                        "interpolation points per step (1 = plain STE; >1 = IG-under-intervention). "
                        "Routed to scores through the chosen STE, so works with hard_topk (Adam) "
                        "and hard_topk_identity (SGD).")
    p.add_argument("--nodes", default="mlp", choices=["mlp", "mlp+attn_dim", "mlp+attn_head", "node", "mlp_tied", "mlp_span", "mlp+attn_span", "mlp+attn_head_span", "mlp_sae_span", "resid_sae_span", "das_mlp_span", "das_resid_span"])
    p.add_argument("--variant", default="hard_topk",
                   choices=["topk", "hard_topk", "hard_topk_identity"])  # build_mask gate
    p.add_argument("--mode", default="sufficient", choices=["sufficient", "necessary", "joint"])
    p.add_argument("--loss", default="logit_diff", choices=list(LOSS_CHOICES),
                   help="training loss (see learning_to_attribute.losses): logit_diff, ce, "
                        "logit, prob (bounded), hinge (--hinge-margin), acc (soft-0-1, --acc-temp)")
    p.add_argument("--hinge-margin", type=float, default=2.0, help="margin (logits) for --loss hinge")
    p.add_argument("--acc-temp", type=float, default=1.0, help="temperature for --loss acc (smaller=sharper)")
    p.add_argument("--sae-repo", default=None, help="Llama-Scope SAE repo (auto: LXM for mlp_sae_span, LXR for resid_sae_span)")
    p.add_argument("--sae-dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--das-dim", type=int, default=None, help="DAS rotation subspace rank (default d_model)")
    p.add_argument("--das-lr", type=float, default=1e-3, help="lr for the DAS rotation params")
    p.add_argument("--das-optimizer", default="adam", choices=["adam", "sgd"], help="optimizer for the DAS rotation (separate from --optimizer for scores)")
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--k-schedule", default="log",
                   choices=["uniform", "log", "adaptive_log", "log_both"])
    p.add_argument("--fixed-k-frac", type=float, default=None,
                   help="MAttr ablation: train the mask at a single FIXED k = frac*total nodes "
                        "every step (overrides --k-schedule sampling). e.g. 0.1 = 10%% of nodes.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n_iters", type=int, default=30)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-examples", type=int, default=100)
    p.add_argument("--grad-examples", type=int, default=None,
                   help="# examples in the IG/IxG attribution batch (default: --eval-examples). "
                        "Lower for long-prompt tasks (arc): the batch is captured with grad for "
                        "all layers at once, so long seqs OOM. Independent of the eval-sweep size.")
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

    if args.dataset == "causalgym":
        train = CGDataset(args.task, tok, n=2000, seed=0)
        test = CGDataset(args.task, tok, n=400, seed=1)
    elif args.dataset == "mib":
        # MIB tasks (arc_easy, ...) via HFEAPDataset: (clean, corrupted, [base_id, source_id]),
        # length-matched per example but variable across examples -> node substrate only.
        import sys as _sys
        _sys.path.insert(0, "MIB-circuit-track")
        from MIB_circuit_track.dataset import HFEAPDataset
        hf_task, name_full = f"mib-bench/{args.task}", MODEL_FULLNAMES[args.model]
        train = HFEAPDataset(hf_task, tok, split="train", task=args.task, model_name=name_full)
        test = HFEAPDataset(hf_task, tok, split="validation", task=args.task, model_name=name_full)
    else:
        train = SVADataset(args.task, tok, split="train")
        test = SVADataset(args.task, tok, split="test")

    # seq_len for the per-position node layout: use the modal clean-prompt token length; the
    # train/eval loops only sample pairs of exactly this length (so the mask indices line up).
    lens = Counter(tok(train[i][0], return_tensors="pt").input_ids.shape[1] for i in range(min(400, len(train))))
    seq_len = lens.most_common(1)[0][0]
    logger.info("seq_len=%d (modal clean length; %s)", seq_len, dict(lens))

    SAE = args.nodes in ("mlp_sae_span", "resid_sae_span")
    DAS = args.nodes in ("das_mlp_span", "das_resid_span")
    SPAN = args.nodes in ("mlp_span", "mlp+attn_span", "mlp+attn_head_span") or SAE or DAS
    # node is position-agnostic (mask broadcasts over positions), so it -- like span mode --
    # does not need a fixed seq_len; both skip the modal-length filter (VARLEN).
    VARLEN = SPAN or args.nodes == "node"
    corrupt_topk = args.mode == "necessary"
    hooker = LlamaAttributionHooks(hf, args.nodes, seq_len=seq_len,
                                   sufficient=corrupt_topk, include_input=args.include_input,
                                   num_spans=(NUM_SPANS if SPAN else None))
    if SAE:
        from learning_to_attribute.sae_loader import load_llama_scope_saes
        comp = "M" if args.nodes == "mlp_sae_span" else "R"
        repo = args.sae_repo or f"fnlp/Llama3_1-8B-Base-LX{comp}-8x"
        sdt = torch.float32 if args.sae_dtype == "float32" else torch.bfloat16
        logger.info("Loading %d Llama-Scope SAEs (%s, component=%s, %s)...",
                    hooker.num_layers, repo, comp, sdt)
        hooker.set_saes(load_llama_scope_saes(repo, hooker.num_layers, device, dtype=sdt, component=comp))
    if DAS:
        logger.info("Creating %d DAS rotations (d_model=%d -> rot-dim=%d)...",
                    hooker.num_layers, hooker.hidden_size, args.das_dim or hooker.hidden_size)
        hooker.set_das(args.das_dim, device=device, dtype=torch.float32)
    total = hooker.total
    logger.info("Nodes (%s): %s", args.nodes, hooker.describe())
    if SPAN:
        logger.info("Span mode: %d content spans, variable length (no seq_len filter)", NUM_SPANS)
    hooker.register_hooks()

    def set_span(cleans, corrupteds):
        """Set the per-batch base/source span-last positions on the hooker (span mode)."""
        if not SPAN:
            return
        hooker.span_last = torch.tensor([SPAN_LAST[c] for c in cleans], device=device)
        hooker.span_last_src = torch.tensor([SPAN_LAST[c] for c in corrupteds], device=device)

    def sample_batch(ds, B, n):
        cl, co, ci, ii = [], [], [], []
        tries = 0
        while len(cl) < B and tries < B * 20:
            tries += 1
            clean, corr, lab = ds[random.randint(0, n - 1)]
            if not VARLEN:  # per-position layout needs a fixed length; span/node take any length
                a = tok(clean, return_tensors="pt").input_ids
                b = tok(corr, return_tensors="pt").input_ids
                if a.shape[1] != seq_len or b.shape[1] != seq_len:
                    continue
            elif not SPAN:  # node: interpolation needs clean/corrupted the SAME length per pair
                if tok(clean, return_tensors="pt").input_ids.shape[1] != \
                   tok(corr, return_tensors="pt").input_ids.shape[1]:
                    continue
            cl.append(clean); co.append(corr); ci.append(lab[0]); ii.append(lab[1])
        return cl, co, ci, ii

    def forward_last(cleans, corrupteds, ci, ii, mask, sufficient):
        """Run the masked forward; return (last-token logits [B,vocab], base idx, source idx)."""
        bt = tok(cleans, return_tensors="pt", padding=True).to(device)
        st = tok(corrupteds, return_tensors="pt", padding=True).to(device)
        last = bt.attention_mask.sum(1) - 1
        hooker.cache_cf_activations(st.input_ids)
        set_span(cleans, corrupteds)
        old_suf = hooker.sufficient; hooker.sufficient = sufficient
        hooker.mask = mask
        logits = hf(bt.input_ids, attention_mask=bt.attention_mask).logits.float()
        hooker.sufficient = old_suf
        B = len(cleans)
        ll = logits[torch.arange(B, device=device), last]
        return ll, torch.tensor(ci, device=device), torch.tensor(ii, device=device)

    def forward_logit_diff(cleans, corrupteds, ci, ii, mask, sufficient):
        ll, cor, inc = forward_last(cleans, corrupteds, ci, ii, mask, sufficient)
        ar = torch.arange(ll.shape[0], device=device)
        return ll[ar, cor] - ll[ar, inc]

    n_train = len(train)
    if args.fixed_k_frac is not None:
        k_sampler = FixedK(max(1, round(args.fixed_k_frac * total)))
        logger.info("Fixed-k training: k=%d (%.1f%% of %d)", k_sampler.k,
                    100 * args.fixed_k_frac, total)
    elif args.k_schedule == "adaptive_log":
        k_sampler = AdaptiveLogK(total)
    else:
        k_sampler = None

    IG_STEPS = args.mattr_ig_steps

    def loss_fn(mask):
        cl, co, ci, ii = sample_batch(train, args.train_batch_size, n_train)
        if not cl:
            return None
        step_cause = resolve_direction(args.mode, corrupt_topk)   # joint -> per-step coin flip
        if IG_STEPS <= 1:
            ll, cor, inc = forward_last(cl, co, ci, ii, mask, sufficient=step_cause)
            if k_sampler is not None:  # feed adaptive-k sampler the decided fraction at this k
                ar = torch.arange(ll.shape[0], device=device)
                with torch.no_grad():
                    dec = (ll[ar, cor] > ll[ar, inc]).float().mean().item()
                k_sampler.observe(dec)
            return attribution_loss(args.loss, ll, cor, inc, corrupt_topk=step_cause,
                                    hinge_margin=args.hinge_margin, acc_temp=args.acc_temp)
        # ---- MAttr-IG: integrate dL/dmask over the baseline(CF)->clean mask path ----
        # effective mask alpha*m_hard makes activations cf + alpha*m*(clean-cf): alpha=0 is the
        # all-baseline circuit, alpha=1 the top-k intervention. a_ig_j = mean_alpha dL/d(mask_j)
        # is the IG attribution of node j; the surrogate (a_ig * mask).sum() re-routes it to the
        # scores through mask's STE (identity or sigmoid), so no trainer change is needed.
        bt = tok(cl, return_tensors="pt", padding=True).to(device)
        st = tok(co, return_tensors="pt", padding=True).to(device)
        last = bt.attention_mask.sum(1) - 1
        hooker.cache_cf_activations(st.input_ids)
        set_span(cl, co)
        B = len(cl); ar = torch.arange(B, device=device)
        cor = torch.tensor(ci, device=device); inc = torch.tensor(ii, device=device)
        m_hard = mask.detach()
        a_ig = torch.zeros_like(m_hard); L1 = None
        old_suf = hooker.sufficient; hooker.sufficient = step_cause
        for j in range(1, IG_STEPS + 1):
            mm = (float(j) / IG_STEPS * m_hard).requires_grad_(True)
            hooker.mask = mm
            logits = hf(bt.input_ids, attention_mask=bt.attention_mask).logits.float()
            Lj = attribution_loss(args.loss, logits[ar, last], cor, inc, corrupt_topk=step_cause,
                                  hinge_margin=args.hinge_margin, acc_temp=args.acc_temp)
            a_ig = a_ig + torch.autograd.grad(Lj, mm)[0]
            if j == IG_STEPS:
                L1 = Lj.detach()
        hooker.sufficient = old_suf
        a_ig = a_ig / IG_STEPS
        surrogate = (a_ig.detach() * mask).sum()   # d/dscores = STE(a_ig); value carries L(alpha=1)
        return surrogate - surrogate.detach() + L1

    # ---- eval helpers (defined pre-training so an optional train-probe can call them) ----
    @torch.no_grad()
    def eval_metrics(examples, mask, sufficient):
        xc, xco, xci, xii = examples
        LB, LS, PB, PS = [], [], [], []
        for s in range(0, len(xc), 20):
            bt = tok(xc[s:s+20], return_tensors="pt", padding=True).to(device)
            st = tok(xco[s:s+20], return_tensors="pt", padding=True).to(device)
            last = bt.attention_mask.sum(1) - 1
            hooker.cache_cf_activations(st.input_ids)
            set_span(xc[s:s+20], xco[s:s+20])
            old = hooker.sufficient; hooker.sufficient = sufficient; hooker.mask = mask.to(device)
            logits = hf(bt.input_ids, attention_mask=bt.attention_mask).logits.float()
            hooker.sufficient = old
            B = bt.input_ids.shape[0]; ar = torch.arange(B, device=device)
            ll = logits[ar, last]; probs = ll.softmax(-1)
            cor = torch.tensor(xci[s:s+20], device=device); inc = torch.tensor(xii[s:s+20], device=device)
            LB.append(ll[ar, cor]); LS.append(ll[ar, inc]); PB.append(probs[ar, cor]); PS.append(probs[ar, inc])
        lb = torch.cat(LB); ls = torch.cat(LS); pb = torch.cat(PB); ps = torch.cat(PS)
        ld = lb - ls
        return {"logit_diff": ld.mean().item(),
                "p_base": pb.mean().item(), "p_source": ps.mean().item(),
                "logit_base": lb.mean().item(), "logit_source": ls.mean().item(),
                "ce_base": (-pb.clamp_min(1e-9).log()).mean().item(),
                "ce_source": (-ps.clamp_min(1e-9).log()).mean().item(),
                "acc_base": (lb > ls).float().mean().item(),
                "acc_source": (ls > lb).float().mean().item(),
                "log_odds_ratio": ld.mean().item(),
                "odds_ratio": float(torch.exp(ld.mean()))}

    sparsities = sorted(set(float(10 ** x) for x in np.linspace(np.log10(1.0/total), 0.0, 24)))
    xs = [s * total for s in sparsities]

    def auc_of(ys):
        lx = np.log10(xs); ya = np.asarray(ys, float)
        return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))

    def summarize(scores_, examples):
        """Full metric suite (both directions) for a ranking on a set of examples."""
        FM = eval_metrics(examples, torch.ones(total), sufficient=False)["logit_diff"]
        F0 = eval_metrics(examples, torch.zeros(total), sufficient=False)["logit_diff"]
        denom = (FM - F0) or 1e-9
        def metrics_at(mask, sufficient):
            m = eval_metrics(examples, mask, sufficient)
            m["faithfulness"] = (m["logit_diff"] - F0) / denom
            return m
        iso = sparsity_sweep(scores_, total, sparsities, lambda hm: metrics_at(hm, False),
                             device=device, include_random=False)["learned"]
        cause = sparsity_sweep(scores_, total, sparsities, lambda hm: metrics_at(hm, True),
                               device=device, include_random=False)["learned"]
        acc = iso["acc_base"]
        kstar = lambda thr: next((float(x) for x, a in zip(xs, acc) if a >= thr), None)
        return dict(F_clean=FM, F_patch=F0, faith_auc=auc_of(iso["faithfulness"]),
                    faith_max=max(iso["faithfulness"]), faithfulness=iso["faithfulness"],
                    cause_auc=auc_of(cause["faithfulness"]), cause_curve=cause["faithfulness"],
                    cause_psrc_auc=auc_of(cause["p_source"]),
                    cause_accsrc_auc=auc_of(cause["acc_source"]),
                    acc_auc=auc_of(acc), kstar_50=kstar(0.5), kstar_90=kstar(0.9),
                    iso_metrics=iso, cause_metrics=cause)

    # optional training-time probe: full metric suite on a FIXED tiny train subset every N steps
    # (unconfounded by the per-step budget k, unlike the raw train loss).
    train_eval_log = []
    on_step_cb = None
    if args.method == "mattr" and args.train_eval_every > 0:
        probe_ex = sample_batch(train, args.train_eval_examples, n_train)
        def on_step_cb(step, k, loss, live_scores):
            if step % args.train_eval_every == 0 or step == args.steps - 1:
                m = summarize(live_scores.detach().cpu(), probe_ex)
                train_eval_log.append({"step": step, **{kk: m[kk] for kk in
                    ("acc_auc", "faith_auc", "kstar_50", "cause_accsrc_auc", "F_clean", "F_patch")}})
                logger.info("  [probe %4d] acc_auc=%.3f faith_auc=%.3f k*=%s",
                            step, m["acc_auc"], m["faith_auc"], m["kstar_50"])

    train_loss_log = None
    if args.method == "random":
        scores = torch.randn(total, device=device)   # random-ranking baseline (seeded)
    elif args.method in ("ixg", "relp", "ig", "conductance"):
        cond = args.method == "conductance"
        scores = gradient_scores(hf, hooker, train, seq_len, total, tok, device,
                                 n_examples=(args.grad_examples or args.eval_examples),
                                 relp=(args.method == "relp"),
                                 ig_steps=args.ig_steps if args.method in ("ig", "conductance") else 1,
                                 loss=args.loss, hinge_margin=args.hinge_margin, acc_temp=args.acc_temp,
                                 conductance=cond)
    else:
        logger.info("Training %d steps (%s gate, %s, %s, k=%s)...", args.steps, args.variant,
                    args.mode, args.optimizer, args.k_schedule)
        # DAS jointly learns the rotation matrices (a second param group) alongside scores.
        das_params = hooker.das_parameters() if DAS else None
        res = learn_scores(total, loss_fn, steps=args.steps, variant=args.variant,
                           k_schedule=args.k_schedule, T=args.T, n_iters=args.n_iters,
                           lr=args.lr, optimizer=args.optimizer, use_bias=False, device=device,
                           k_sampler=k_sampler, extra_params=das_params, lr_extra=args.das_lr,
                           extra_optimizer=(args.das_optimizer if DAS else None), on_step=on_step_cb)
        scores = res.scores.detach()
        train_loss_log = res.loss_log
        if isinstance(k_sampler, AdaptiveLogK):
            logger.info("adaptive-k final frontier: k_max=%.0f (%.2f%% of %d)",
                        math.exp(k_sampler.kmax_log),
                        100 * math.exp(k_sampler.kmax_log) / total, total)

    # ---- Phase 2: sparsity sweep, BOTH directions, counterfactual (patch) ablation ----
    #   iso  (sufficiency): keep top-k CLEAN, corrupt the complement  -> recovery curve
    #   cause(necessity):   corrupt top-k, keep the complement clean  -> breakage curve
    # Same top-k ranking; only the hooker `sufficient` flag flips. Complement/top-k are ablated
    # to each example's own counterfactual (patch), matching training (mean-abl deferred).
    ec, eco, eci, eii = [], [], [], []
    for i in range(len(test)):
        clean, corr, lab = test[i]
        if not VARLEN:  # span/node evaluate variable-length pairs
            if tok(clean, return_tensors="pt").input_ids.shape[1] != seq_len: continue
            if tok(corr, return_tensors="pt").input_ids.shape[1] != seq_len: continue
        elif not SPAN:   # node: clean/corrupted must match length (interpolation alignment)
            if tok(clean, return_tensors="pt").input_ids.shape[1] != \
               tok(corr, return_tensors="pt").input_ids.shape[1]: continue
        ec.append(clean); eco.append(corr); eci.append(lab[0]); eii.append(lab[1])
        if len(ec) >= args.eval_examples: break
    logger.info("Eval on %d test pairs (%s)", len(ec), "variable len" if VARLEN else f"len={seq_len}")

    S = summarize(scores, (ec, eco, eci, eii))   # eval_metrics/summarize defined above (pre-training)
    FM, F0 = S["F_clean"], S["F_patch"]
    faith_auc, acc_auc, kstar_50 = S["faith_auc"], S["acc_auc"], S["kstar_50"]
    logger.info("F(clean)=%.3f  F(patch)=%.3f", FM, F0)
    logger.info("acc_auc=%.3f  k*(>0.5)=%s  k*(>0.9)=%s", acc_auc, kstar_50, S["kstar_90"])
    hooker.remove_hooks()

    out = dict(task=args.task, model=args.model, nodes=args.nodes, variant=args.variant,
               mode=args.mode, optimizer=args.optimizer, k_schedule=args.k_schedule,
               total=total, seq_len=seq_len, n_nodes=xs, **S)   # S carries all metric curves+AUCs
    out["intermediate_size"] = hooker.intermediate_size
    out["hidden_size"] = hooker.hidden_size
    out["num_layers"] = hooker.num_layers
    out["loss"] = args.loss
    out["loss_log"] = train_loss_log
    out["train_eval_log"] = train_eval_log
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    tag = args.method if args.method != "mattr" else f"{args.mode}_{args.variant}_{args.optimizer}"
    if args.method == "random":
        tag = f"random_s{args.seed}"
    if args.loss != "logit_diff":   # encode the loss target for BOTH mattr and gradient methods
        tag += f"_{args.loss}"
    if args.method == "mattr" and args.mattr_ig_steps > 1:
        tag += f"_ig{args.mattr_ig_steps}"
    if args.method == "mattr" and args.fixed_k_frac is not None:
        tag += f"_fixedk{int(round(args.fixed_k_frac * 100))}"
    if args.method == "mattr" and args.fixed_k_frac is None and args.k_schedule == "uniform":
        tag += "_uniformk"
    if args.method == "mattr" and args.k_schedule == "adaptive_log":
        tag += "_adaptivek"
    if args.method == "mattr" and args.k_schedule == "log_both":
        tag += "_logboth"
    if args.method == "mattr" and args.train_batch_size != 8:
        tag += f"_bs{args.train_batch_size}"
    if args.method == "mattr" and args.steps != 2000:
        tag += f"_s{args.steps}"
    if args.method == "mattr" and args.loss == "acc" and args.acc_temp != 1.0:
        tag += f"_t{str(args.acc_temp).replace('.', '')}"
    fn = outdir / f"{args.task}_{args.model}_{args.nodes.replace('+','-')}_{tag}.json"
    torch.save(scores.cpu(), fn.with_suffix(".scores.pt"))
    json.dump(out, open(fn, "w"), indent=2)
    logger.info("iso/faith AUC=%.3f (fmax %.3f) | cause AUC=%.3f | total=%d -> %s",
                S["faith_auc"], S["faith_max"], S["cause_auc"], total, fn)


if __name__ == "__main__":
    main()

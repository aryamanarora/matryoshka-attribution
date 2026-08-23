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

from learning_to_attribute import learn_scores, sparsity_sweep, wandb_util
from learning_to_attribute.edge_pruning import (
    learn_scores_edge_pruning, learn_scores_sigmoid_mask)
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


ARITH_DIR = "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B"


class ArithDataset:
    """goodfire-ai/arithmetic-wild task as a fixed list of (clean, corrupted, [base_id, source_id]).

    Drop-in for SVADataset. The upstream release pairs each base with its counterfactual by
    index, so train/test are disjoint index ranges rather than two seeds -- with 1.6-4k pairs
    and sampling with replacement, two seeds would overlap heavily.

    Two task-specific wrinkles, both handled here rather than downstream:
      * `hours` answers are multi-token ("04:00" -> ["04", ":", "00"]). We score the FIRST
        token, which is the only one that varies with the answer -- ":" and "00" are constant,
        so a logit diff on them is identically zero.
      * base and counterfactual answers coincide by chance in 1-14% of pairs (highest for
        weekdays, which has only 7 possible answers). Those pairs have a zero logit diff in
        either direction and are dropped, not left to contribute a null gradient.
    """
    def __init__(self, task, tokenizer, split="train", frac=0.8, data_dir=ARITH_DIR):
        from learning_to_attribute.data.arithmetic_wild import ArithmeticWildDataset
        ds = ArithmeticWildDataset(task, data_dir)
        n = len(ds.bases)
        idx = range(0, int(n * frac)) if split == "train" else range(int(n * frac), n)
        self.recs, self.dropped = [], 0
        for i in idx:
            b, c = ds.bases[i], ds.cfs[i]
            bid = tokenizer.encode(b["raw_output"], add_special_tokens=False)[0]
            sid = tokenizer.encode(c["raw_output"], add_special_tokens=False)[0]
            if bid == sid:
                self.dropped += 1
                continue
            self.recs.append((b["raw_input"], c["raw_input"], [bid, sid]))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        return self.recs[i]


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
                    loss="logit_diff", hinge_margin=2.0, acc_temp=1.0, conductance=False, attnlrp=False,
                    mc=False, mc_seed=0):
    """Closed-form gradient attribution (IxG = grad x delta) over the hooker's node layout.

    Captures the clean activation at each node module (down_proj / o_proj input) with a
    forward-pre-hook (retain_grad), runs a clean forward + logit-diff backward, and scores each
    node by g . (clean - patch), summed over a batch. relp=True applies the RelP modified
    backward first (LN-freeze + MLP gate rule + QK-detach); attnlrp=True applies AttnLRP's
    instead (LN-freeze + MLP gate rule + half-rule on the QK/OV matmuls, softmax kept).

    mc=True is "stepless IG": draw alpha ~ U(0,1) PER EXAMPLE instead of walking the fixed grid
    alpha = s/ig_steps. The grid below is a LEFT-endpoint Riemann sum over [0,1) -- it contains
    the clean endpoint (alpha=0) and omits the patch one -- so at ig_steps=1 it degenerates to
    the single point alpha=0 and IG *is* IxG (that is exactly what --method ixg computes). The
    MC estimator is unbiased for the same integral at EVERY ig_steps, including 1, at identical
    cost: one forward+backward per draw either way. So `--method mc_ig --ig-steps 1` against
    `--method ixg` is a compute-matched contrast whose only difference is where alpha is placed.

    Alpha is [B,1,1] so it broadcasts over (pos, d_model) -- B independent draws for the price
    of one forward, and since scores sum over the batch before anything else the estimator error
    falls like 1/sqrt(n_examples), not 1/sqrt(n_batches).

    This mirrors get_scores_eap_ig_mc in MIB-circuit-track/EAP-IG/src/eap/attribute_node.py; the
    two harnesses must stay in step or the SVA and MIB stepless-IG numbers stop being the same
    estimator. Note the ALPHA CONVENTION IS REVERSED between them (here alpha=0 is clean and
    alpha=1 is patch; there alpha=1 is clean) -- U(0,1) is symmetric so the estimator is
    identical, but do not copy an alpha expression across without checking which end is which.
    """
    if hooker.mask_type in ("mlp_sae_span", "resid_sae_span", "das_mlp_span", "das_resid_span"):
        raise NotImplementedError("gradient attribution not supported for SAE/DAS nodes; use --method mattr")
    assert not (relp and attnlrp), "relp and attnlrp are alternative backward rule sets"
    modified_bwd = relp or attnlrp
    if modified_bwd:
        from learning_to_attribute.grad_attribution import install_attnlrp, install_relp, revert_relp
        (install_attnlrp if attnlrp else install_relp)(hf)
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
    if hooker.zero_ablation:
        # Match the intervention these scores will be EVALUATED under: the ablated value is 0,
        # so the endpoint delta is (clean - 0) = clean. This is not a cosmetic change -- it turns
        # IxG into plain Gradient x Input and IG into the textbook zero-baseline IG, which are
        # different estimators from the counterfactual-baseline ones, not the same method rescored.
        patch_acts = {k: torch.zeros_like(v) for k, v in patch_acts.items()}

    # embeddings for the IG path (interpolate clean->patch input embedding, downstream live)
    # CPU generator: the alpha stream then depends only on mc_seed and the batch shape, not on
    # how much of the global torch RNG the rest of the run has already consumed. Without this a
    # seed replicate would silently stop being a clean replicate the moment anything upstream
    # (dataset shuffling, a random baseline) changed its own draw count.
    gen = torch.Generator(device="cpu"); gen.manual_seed(mc_seed)

    emb_override = None
    if ig_steps > 1 or conductance or hooker.include_input or mc:
        cap = {}
        h = hf.model.embed_tokens.register_forward_hook(lambda m, i, o: cap.__setitem__("e", o.detach()))
        with torch.no_grad(): hf(bid, attention_mask=bam); ec = cap["e"]
        with torch.no_grad(): hf(pt.input_ids, attention_mask=pt.attention_mask); ep = cap["e"]
        h.remove()
        if hooker.zero_ablation:
            ep = torch.zeros_like(ep)   # IG integrates from the ZERO embedding, as at the nodes

    def input_node_effect():
        # score for the input-embedding node (index 0 when include_input): grad(emb).(clean-patch),
        # averaged over the IG path. The input embedding is interpolated LINEARLY, so its IG and
        # conductance coincide; ixg (ig_steps=1) is grad at the clean embedding.
        S = ig_steps if ig_steps > 1 else 1
        g_acc = torch.zeros_like(ec)
        for step in range(1, S + 1):
            # MC draws alpha per example here too. If it did not, the input node would be the one
            # unit in the circuit still scored off the grid while every other unit was scored by
            # MC -- a mixed estimator, and specifically one where the input node is the unit most
            # likely to be mis-ranked (it is top-1 on the SVA depth artifact).
            if mc:
                a = torch.rand(ec.shape[0], 1, 1, generator=gen).to(ec)
                eo = (ep + a * (ec - ep)).detach().requires_grad_(True)
            else:
                eo = (ep + (step / S) * (ec - ep)).detach().requires_grad_(True)  # step=S -> clean
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
        if modified_bwd:
            revert_relp(hf)
        return scores.to(device)

    grad_acc = {k: torch.zeros_like(v) for k, v in clean_acts.items()}
    # The grid is LEFT-endpoint over [0,1): alpha in {0, 1/m, ..., (m-1)/m}, clean end included,
    # patch end excluded. At m=1 that is the single point alpha=0, i.e. the gradient at the clean
    # input -- IxG, not an integral estimate. MC replaces the grid with ig_steps independent
    # U(0,1) draws per example, which IS an integral estimate at the very same m.
    n_draws = ig_steps if (mc or ig_steps > 1) else 1
    for step in range(n_draws):
        if mc:
            a = torch.rand(ec.shape[0], 1, 1, generator=gen).to(ec)
            emb_override = (1 - a) * ec + a * ep
        elif ig_steps > 1:
            emb_override = (1 - step / ig_steps) * ec + (step / ig_steps) * ep
        store_g, logits = capture(bid, bam, True, embed_override=emb_override)
        metric_of(logits).backward()
        for k in grad_acc:
            grad_acc[k] += store_g[k].grad
    for k in grad_acc:
        grad_acc[k] /= n_draws

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
        if modified_bwd:
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
        if modified_bwd:
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
    if modified_bwd:
        revert_relp(hf)
    return scores.to(device)


def run_tag(args):
    """The method half of the output filename: `<task>_<model>_<nodes>_<TAG>.json`.

    Factored out of the write at the end of main() so wandb can NAME the run before training
    starts. Beware: the tag deliberately encodes only knobs that change the *identity* of the
    circuit, so two runs differing solely in --steps/--lr/etc. beyond the defaults handled
    below collide on disk -- probe sweeps must use a separate --output dir.
    """
    tag = args.method if args.method != "mattr" else f"{args.mode}_{args.variant}_{args.optimizer}"
    if args.method == "random":
        tag = f"random_s{args.seed}"
    if args.method == "mc_ig":
        # BOTH the draw count and the seed are part of the identity, unlike every other method
        # here. Two things force it. (1) --ig-steps is NOT otherwise encoded in a tag -- `ig` at
        # 5 and at 30 steps already collide on disk -- and mc_ig's headline claim is specifically
        # about m=1, so an unlabelled m=10 run sitting in the same filename would silently
        # restate a 10x-cost result as the free one. (2) The seed IS the error bar: replicates
        # differing only in --seed are how this estimator's noise floor gets measured, so they
        # must not overwrite each other the way MIB's would have without a per-seed circuit dir.
        tag = f"mc_ig_m{args.ig_steps}_s{args.seed}"
    if args.method == "edge_pruning":   # e.g. eprun_s090 -- budget is part of the identity
        tag = f"eprun_s{int(round(args.target_sparsity * 100)):03d}"
    if args.method == "sigmoid_mask":
        # e.g. sig_lr0.3_l16.0 -- lr and the penalty are the two knobs that decide the circuit,
        # so both are part of the identity, spelled the way the MIB dirs spell them
        # (results/eprun_node_ld_sig_lr0.3_l16.0) so the two harnesses' runs read alike.
        # Plain str() of the float, NOT :g -- str(6.0) is "6.0" but f"{6.0:g}" is "6", and
        # "sig_lr0.3_l16" reads as l1=16 as easily as l1=6. It also keeps the spelling identical
        # to the MIB dirs (eprun_node_ld_sig_lr0.3_l16.0), which is what lets a reader match a
        # run across the two harnesses by name.
        tag = f"sig_lr{args.lr}"
        if args.l1_coeff:
            tag += f"_l1{'logit' if args.l1_target == 'logit' else ''}{args.l1_coeff}"
    if args.loss != "logit_diff":   # encode the loss target for BOTH mattr and gradient methods
        tag += f"_{args.loss}"
    if args.ablation != "patch":
        # applies to EVERY method including the gradient ones, so it goes here rather than in a
        # mattr-only branch -- a zero-ablation IG is a different circuit from a patched IG and
        # must not overwrite it.
        tag += f"_{args.ablation}abl"
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
    return tag


def wandb_init(args, tag):
    """Start the run. Named `run_tag`, i.e. exactly the output filename's method half, so a
    chart can be matched back to its json without a lookup table."""
    return wandb_util.init(
        args.dataset,
        f"{args.task}_{args.model}_{args.nodes.replace('+', '-')}_{tag}",
        vars(args), project=args.wandb_project, entity=args.wandb_entity,
        enabled=args.wandb, group=f"{args.task}/{args.nodes}", job_type=args.method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llama3", choices=list(MODEL_FULLNAMES))
    p.add_argument("--task", required=True)            # sva: nounpp|rc|simple|within_rc ; causalgym: e.g. npi_any_subj-relc
    p.add_argument("--dataset", default="sva", choices=["sva", "causalgym", "mib", "arith"])
    p.add_argument("--method", default="mattr",
                   choices=["mattr", "ixg", "relp", "attnlrp", "ig", "mc_ig", "conductance",
                            "random", "edge_pruning", "sigmoid_mask"])
    # Node/Edge Pruning (Bhaskar et al., 2024) on this harness: hard-concrete gates + a
    # Lagrangian L0 budget instead of MAttr's top-k. It takes the SAME loss_fn as MAttr, so
    # --loss still selects the objective and the only thing that differs is how the mask is
    # parameterized and constrained -- which is the comparison the figure is about. `total`
    # here is the substrate size (MLP neurons, or neurons + attn heads), not MIB's ~156 nodes,
    # so the budget is on a very different absolute scale than results/eprun_node_s*.
    p.add_argument("--target-sparsity", type=float, default=0.9,
                   help="edge_pruning: fraction of units the L0 Lagrangian anneals to PRUNING.")
    # sigmoid_mask = the pyvene SigmoidMaskIntervention baseline the MIB tables show as DBM:
    # deterministic sigmoid(mask/temp), temperature annealed 50 -> 0.1, no L0 term. Same
    # loss_fn and the same step budget as MAttr and Node Pruning, so once again the mask
    # parameterization is the only thing that varies. It has no --target-sparsity: the anneal
    # controls how BINARY the gate is, not how sparse, and sparsity comes from --l1-coeff (or,
    # at 0, from the sweep ranking the logits like any other score).
    p.add_argument("--l1-coeff", type=float, default=0.0,
                   help="sigmoid_mask: L1 sparsity penalty weight. 0 = the pyvene library's own "
                        "unpenalised recipe; >0 = its tutorial's penalised one.")
    p.add_argument("--l1-target", default="gate", choices=["gate", "logit"],
                   help="sigmoid_mask: 'gate' penalises mean gate value (an L0 relaxation, "
                        "normalised by substrate size); 'logit' is pyvene's tutorial term "
                        "coeff*||mask||_1, which pulls gates toward 0.5 rather than 0.")
    p.add_argument("--ig-steps", type=int, default=10, help="IG integration steps (input-embedding path)")
    p.add_argument("--train-eval-every", type=int, default=200,
                   help="Mask-learning methods: every N steps, run the FULL eval-metric suite on "
                        "a fixed tiny train subset and log it. The train LOSS is measured at a k "
                        "that moves over training, so it is not comparable across steps; these "
                        "AUCs integrate over the whole k grid and are. 0 = off.")
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
    p.add_argument("--ablation", default="patch", choices=["patch", "zero"],
                   help="what the ablated units are set to. patch (default) = the cached SOURCE "
                        "activation from the counterfactual prompt; zero = 0. This is a property "
                        "of the whole run: MAttr TRAINS through the same intervention it is "
                        "scored with, and the faithfulness endpoints F_clean/F_patch are "
                        "recomputed under it, so the two settings are not comparable run-for-run "
                        "-- only method RANKINGS within a setting are. It also redefines the "
                        "gradient baselines: with a zero baseline IxG becomes Gradient x Input "
                        "and IG becomes textbook zero-baseline IG.")
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
    wandb_util.add_args(p)     # --no-wandb / --wandb-project / --wandb-entity; ON by default
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); torch.manual_seed(args.seed)
    tag = run_tag(args)
    wb = wandb_init(args, tag)

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
    elif args.dataset == "arith":
        # arithmetic-wild has no span schema built here, so the per-span substrates would
        # silently fall back to a wrong NUM_SPANS; refuse them explicitly.
        assert args.nodes not in ("mlp_span", "mlp+attn_span", "mlp+attn_head_span",
                                  "mlp_sae_span", "resid_sae_span",
                                  "das_mlp_span", "das_resid_span"), \
            f"--dataset arith does not build a span schema; {args.nodes} needs one"
        train = ArithDataset(args.task, tok, split="train")
        test = ArithDataset(args.task, tok, split="test")
        logger.info("arith %s: %d train / %d test pairs (dropped %d/%d with base==cf answer)",
                    args.task, len(train), len(test), train.dropped + test.dropped,
                    len(train) + len(test) + train.dropped + test.dropped)
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
                                   num_spans=(NUM_SPANS if SPAN else None),
                                   zero_ablation=args.ablation == "zero")
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

    # ---- training-time probe: the full metric suite on a FIXED tiny TRAIN subset every N steps.
    # This is the only honest way to watch a mask-learning run converge. The raw train loss is
    # measured at a k that MOVES over training (the log-k schedule samples a new budget every
    # step, and AdaptiveLogK widens its frontier as accuracy rises), so a falling loss curve
    # conflates "the ranking got better" with "this step happened to draw an easier k" -- two
    # runs' losses at step t are not even the same quantity. acc-AUC / faith-AUC integrate over
    # the whole k grid, so they are k-independent and comparable across steps, runs and methods.
    # The subset is TRAIN, and fixed across the run, so the probe is a convergence diagnostic,
    # not a held-out estimate: read it for "has it stopped improving", never as a test number.
    #
    # AND A PLATEAU IS NOT PROOF OF CONVERGENCE. At --train-eval-examples 16 the probe
    # SATURATES: on nounpp/mlp it read 0.685 at step 2000 and 0.690 at 6250 (flat), while the
    # 100-example TEST acc-AUC of the same configuration rose 0.663 -> 0.705 over that span.
    # It caught the large gap it was built for (addition/mlp, +0.09 over the same span) and
    # missed a real +0.04. Treat a rising probe as evidence of under-convergence; treat a flat
    # one as inconclusive, and raise --train-eval-examples before believing it.
    train_eval_log = []
    on_step_cb = None
    # edge_pruning/sigmoid_mask hand back rankable scores from their on_step too (log-alphas and
    # mask logits respectively), so they get the same probe -- it is how we can tell an
    # under-converged L0 anneal from a converged one without waiting for the final sweep.
    if args.method in ("mattr", "edge_pruning", "sigmoid_mask") and (args.train_eval_every > 0
                                                                    or wb is not None):
        probe_ex = (sample_batch(train, args.train_eval_examples, n_train)
                    if args.train_eval_every > 0 else None)
        def on_step_cb(step, k, loss, live_scores):
            if wb is not None:
                wb.log({"train/loss": loss, "train/k": k, "train/k_frac": k / total}, step=step)
            if probe_ex is None:
                return
            if step % args.train_eval_every == 0 or step == args.steps - 1:
                m = summarize(live_scores.detach().cpu(), probe_ex)
                rec = {kk: m[kk] for kk in
                       ("acc_auc", "faith_auc", "kstar_50", "cause_accsrc_auc", "F_clean", "F_patch")}
                train_eval_log.append({"step": step, **rec})
                if wb is not None:
                    wb.log({f"probe/{kk}": v for kk, v in rec.items() if v is not None}, step=step)
                logger.info("  [probe %4d] acc_auc=%.3f faith_auc=%.3f k*=%s",
                            step, m["acc_auc"], m["faith_auc"], m["kstar_50"])

    train_loss_log = None
    if args.method == "random":
        scores = torch.randn(total, device=device)   # random-ranking baseline (seeded)
    elif args.method in ("ixg", "relp", "attnlrp", "ig", "mc_ig", "conductance"):
        cond = args.method == "conductance"
        # mc_ig reads --ig-steps as its NUMBER OF DRAWS, so it must be in this list; the whole
        # point of the arm is --ig-steps 1, which for every other method here means "no path".
        scores = gradient_scores(hf, hooker, train, seq_len, total, tok, device,
                                 n_examples=(args.grad_examples or args.eval_examples),
                                 relp=(args.method == "relp"), attnlrp=(args.method == "attnlrp"),
                                 ig_steps=args.ig_steps if args.method in ("ig", "mc_ig", "conductance") else 1,
                                 loss=args.loss, hinge_margin=args.hinge_margin, acc_temp=args.acc_temp,
                                 conductance=cond,
                                 mc=(args.method == "mc_ig"), mc_seed=args.seed)
    elif args.method == "edge_pruning":
        # Same loss_fn as MAttr -- only the mask parameterization differs (hard-concrete gates
        # under an annealed L0 budget vs top-k). No k_sampler: the budget IS the L0 target, and
        # the returned log-alphas are ranked by the sweep below exactly like any other score.
        logger.info("Edge Pruning: %d steps, target sparsity %.3f over %d units",
                    args.steps, args.target_sparsity, total)
        res = learn_scores_edge_pruning(total, loss_fn, steps=args.steps,
                                        target_sparsity=args.target_sparsity, device=device,
                                        logger=logger, log_every=200, on_step=on_step_cb)
        scores = res.scores.detach()
        train_loss_log = res.loss_log
        kept = res.train_log[-1][1] if getattr(res, "train_log", None) else None
        if kept is not None:
            # The Lagrangian does NOT always bind: at node level on MIB it misses s=0.99 on 10
            # of 11 cells. Log achieved vs requested so an unconverged run is visible here
            # rather than being read off the tag as a budget it never reached.
            logger.info("achieved sparsity %.3f (kept %.1f of %d; requested %.3f)",
                        1 - kept / total, kept, total, args.target_sparsity)
    elif args.method == "sigmoid_mask":
        # Same loss_fn again; the mask is pyvene's deterministic sigmoid gate. The returned
        # scores are the mask LOGITS, monotone in the gate, so the sweep below ranks them
        # exactly like an attribution score -- no rescaling needed.
        logger.info("Sigmoid mask (DBM): %d steps, lr %g, l1 %g (%s) over %d units",
                    args.steps, args.lr, args.l1_coeff, args.l1_target, total)
        res = learn_scores_sigmoid_mask(total, loss_fn, steps=args.steps, lr=args.lr,
                                        l1_coeff=args.l1_coeff, l1_target=args.l1_target,
                                        device=device, logger=logger, log_every=200,
                                        on_step=on_step_cb)
        scores = res.scores.detach()
        train_loss_log = res.loss_log
        kept = res.train_log[-1][1] if getattr(res, "train_log", None) else None
        if kept is not None:
            # Density is an OUTCOME here, not a budget -- unpenalised runs converge dense and
            # even penalised ones are not held to a target. Log it for the same reason Node
            # Pruning logs achieved sparsity: so the number is read off the run, not the tag.
            logger.info("final density %.3f (soft-kept %.1f of %d)", kept / total, kept, total)
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
    # The FULL invocation. The filename tag only encodes knobs that change the circuit's
    # identity, so --lr, --steps (at the default), --seed and the probe settings appear
    # nowhere else -- two runs that differ only in lr write the same filename and the json
    # could not tell you which one you were reading. Additive; nothing parses it yet.
    out["config"] = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))}
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    fn = outdir / f"{args.task}_{args.model}_{args.nodes.replace('+','-')}_{tag}.json"
    torch.save(scores.cpu(), fn.with_suffix(".scores.pt"))
    json.dump(out, open(fn, "w"), indent=2)
    logger.info("iso/faith AUC=%.3f (fmax %.3f) | cause AUC=%.3f | total=%d -> %s",
                S["faith_auc"], S["faith_max"], S["cause_auc"], total, fn)

    if wb is not None:
        # The scalars go in summary (not log) so the run table sorts on them; the sweep curves
        # go in as tables so a chart can be built per-run without re-reading the json.
        wb.summary.update({f"test/{k}": S[k] for k in
                           ("acc_auc", "faith_auc", "cause_auc", "cause_accsrc_auc",
                            "faith_max", "kstar_50", "kstar_90", "F_clean", "F_patch")
                           if S[k] is not None})
        # SummaryDict.update takes a dict POSITIONALLY only -- kwargs raise TypeError.
        wb.summary.update({"total": total, "seq_len": seq_len, "n_eval": len(ec),
                           "json_path": str(fn)})
        try:
            import wandb
            wb.log({"test/sweep": wandb.Table(
                columns=["k", "faith_iso", "acc_iso", "faith_cause", "acc_cause"],
                data=[[float(x), float(a), float(b), float(c), float(d)] for x, a, b, c, d in zip(
                    xs, S["iso_metrics"]["faithfulness"], S["iso_metrics"]["acc_base"],
                    S["cause_metrics"]["faithfulness"], S["cause_metrics"]["acc_base"])])})
        except Exception as exc:                   # noqa: BLE001
            logger.warning("wandb table failed (%s)", exc)
        wb.finish()


if __name__ == "__main__":
    main()

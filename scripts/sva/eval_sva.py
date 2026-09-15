"""Train + evaluate SVA node circuits entirely in-framework (no nnsight, no MIB graph).

Phase 1: learn node scores via LlamaAttributionHooks + build_mask + learn_scores on SVADataset
         (clean/patch minimal pairs, logit-diff objective) — same path as eval_mib.py.
Phase 2: faithfulness sparsity-sweep via the same hooker (keep top-k CLEAN, ablate the rest to
         the patch counterfactual; normalized recovery of the clean logit-diff).

Node sets (--nodes): "mlp" = per-(layer,pos,neuron) MLP acts; "mlp+attn_dim" = that plus
per-(layer,pos,dim) attention pre-out (o_proj input) — i.e. mlp acts per-neuron and attn
pre-out per-dim.
"""
import argparse, json, logging, math, os, random, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import learn_scores, sparsity_sweep, wandb_util, CFActivationCache
from learning_to_attribute.edge_pruning import (
    learn_scores_edge_pruning, learn_scores_sigmoid_mask)
from learning_to_attribute.schedules import AdaptiveLogK, FixedK
from learning_to_attribute.losses import (CLEAN_TARGET_LOSSES, LOSS_CHOICES,
                                           attribution_loss, resolve_direction)
from learning_to_attribute.data import SVADataset, CausalGymDataset
from learning_to_attribute.models import LlamaAttributionHooks


# span-last token positions per string, keyed by the (cleaned) string. Lets the span-mode
# forwards recover each example's per-span node positions without threading them everywhere.
class SpanLast(dict):
    """cleaned string -> [last-token position per content span].

    PER-POSITION MODE makes a "span" a single token position, which turns the *_sae_span
    substrates into the SAE analogue of `mlp`: one score per (layer, position, latent) instead
    of per (layer, content-span, latent). Every prompt then maps to the same 0..seq_len-1 index
    list, so nothing has to be recorded per string -- __missing__ serves it. That is the whole
    trick: the span machinery already gathers/scatters at arbitrary token positions, so
    per-position needs no new mask type, no new hook and no change to llama.py.

    Only usable at a FIXED sequence length, which is why the caller also clears VARLEN -- see
    main(). Outside per-position mode this is an ordinary dict and a missing key is still a
    KeyError, so a genuinely unrecorded span cannot silently read as position 0.
    """
    perpos = None

    def set_per_position(self, seq_len):
        self.perpos = list(range(seq_len))

    def __missing__(self, key):
        if self.perpos is None:
            raise KeyError(key)
        return self.perpos


SPAN_LAST = SpanLast()   # cleaned string -> [last-tok-pos per content span]
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


# goodfire-ai/arithmetic-wild's generated datasets. Resolved like deps.find_mib_path: $L2A_ARITH_DIR,
# then deps/arithmetic-wild (where scripts/setup.sh-era checkouts keep outside repos; the 7 MB
# `datasets/` tree was copied there from Tilde on 2026-09-15), then the Tilde path that was
# hardcoded here until then -- so the older machines keep working and a fresh clone gets a
# clear error naming the fix instead of a Tilde path.
def _arith_dir():
    root = Path(__file__).resolve().parents[2]
    cands = [os.environ.get("L2A_ARITH_DIR"),
             root / "deps" / "arithmetic-wild" / "datasets" / "Llama-3.1-8B",
             "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B"]
    for c in cands:
        if c and Path(c).is_dir():
            return str(c)
    # Not found: return the deps/ location so a later open() fails with THAT path in the error
    # (non-arithmetic tasks never touch it, so this must not raise at import time).
    return str(cands[1])


ARITH_DIR = _arith_dir()


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
                    grad_batch=0,
                    loss="logit_diff", hinge_margin=2.0, acc_temp=1.0, ld_scale=2.0,
                    conductance=False, attnlrp=False, mc=False, mc_seed=0):
    """Closed-form gradient attribution (IxG = grad x delta) over the hooker's node layout.

    Captures the clean activation at each node module (down_proj / o_proj input) with a
    forward-pre-hook (retain_grad), runs a clean forward + logit-diff backward, and scores each
    node by g . (clean - patch), summed over a batch. relp=True applies the RelP modified
    backward first (LN-freeze + MLP gate rule + QK-detach); attnlrp=True applies AttnLRP's
    instead (LN-freeze + MLP gate rule + half-rule on the QK/OV matmuls, softmax kept).

    mc=True is "Expected Gradients": draw alpha ~ U(0,1) PER EXAMPLE instead of walking the fixed grid
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
    two harnesses must stay in step or the SVA and MIB Expected Gradients numbers stop being the same
    estimator. Note the ALPHA CONVENTION IS REVERSED between them (here alpha=0 is clean and
    alpha=1 is patch; there alpha=1 is clean) -- U(0,1) is symmetric so the estimator is
    identical, but do not copy an alpha expression across without checking which end is which.
    """
    if hooker.mask_type in ("das_mlp_span", "das_resid_span"):
        raise NotImplementedError("gradient attribution not supported for DAS nodes; use --method mattr")
    # ---- example-chunking for the NON-SAE branch (the SAE branch chunks internally). Exact,
    # not an approximation: every score below is a SUM over the batch, so summing per-chunk
    # scores reproduces the full-batch result up to fp addition order. This is what makes a
    # compute-matched --grad-examples (thousands, to match MAttr's step budget) runnable at
    # all -- the full-batch path stores 32 layers x [B, P, d] for clean, patch AND grad, which
    # is terabytes at B=5000 on the neuron substrates. Collection happens here (same filters as
    # below) and each chunk recurses with grad_batch=0, BEFORE the RelP/AttnLRP install so the
    # modified backward is installed exactly once per chunk. conductance is excluded: it
    # tracks a running activation trajectory across path steps (node-only and cheap anyway).
    # MC chunks draw alphas from per-chunk seeds (mc_seed + 7919*chunk_idx) -- a different
    # stream than the unchunked run, which is fine for a Monte-Carlo estimator but means a
    # chunked and an unchunked mc run are only statistically, not bitwise, comparable.
    if grad_batch and not hooker.is_sae and not conductance:
        span_ = hooker.mask_type in ("mlp_span", "mlp+attn_span", "mlp+attn_head_span")
        node_ = hooker.mask_type == "node"
        cl_, co_, ci_, ii_ = [], [], [], []
        i_ = 0
        while len(cl_) < n_examples and i_ < len(ds):
            clean_, corr_, lab_ = ds[i_]; i_ += 1
            if not span_ and not node_:
                if tok(clean_, return_tensors="pt").input_ids.shape[1] != seq_len: continue
                if tok(corr_, return_tensors="pt").input_ids.shape[1] != seq_len: continue
            elif node_:
                if tok(clean_, return_tensors="pt").input_ids.shape[1] != \
                   tok(corr_, return_tensors="pt").input_ids.shape[1]: continue
            cl_.append(clean_); co_.append(corr_); ci_.append(lab_[0]); ii_.append(lab_[1])
        acc_scores = None
        for j_, s0 in enumerate(range(0, len(cl_), grad_batch)):
            sub_ds = [(cl_[t], co_[t], [ci_[t], ii_[t]]) for t in range(s0, min(s0 + grad_batch, len(cl_)))]
            part = gradient_scores(hf, hooker, sub_ds, seq_len, total, tok, device,
                                   n_examples=len(sub_ds), relp=relp, ig_steps=ig_steps,
                                   loss=loss, hinge_margin=hinge_margin, acc_temp=acc_temp,
                                   ld_scale=ld_scale, conductance=False, attnlrp=attnlrp,
                                   mc=mc, mc_seed=mc_seed + 7919 * j_, grad_batch=0)
            # metric_of is a BATCH MEAN, so each chunk's per-example weight is 1/chunk_size;
            # rescale by chunk/N so every example carries 1/N exactly as in the full-batch
            # path (the SAE branch's kb/n_total factor, which this wrapper must mirror --
            # without it a ragged last chunk is overweighted by N/chunk_size).
            part = part * (len(sub_ds) / max(1, len(cl_)))
            acc_scores = part if acc_scores is None else acc_scores + part
            torch.cuda.empty_cache()
        logger.info("gradient attribution: %d examples in chunks of %d (%d chunks) x %d draws",
                    len(cl_), grad_batch, j_ + 1, ig_steps if (mc or ig_steps > 1) else 1)
        return acc_scores
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
                                 hinge_margin=hinge_margin, acc_temp=acc_temp,
                                 ld_scale=ld_scale)

    # cached clean & patch node acts (no grad) -> delta
    #
    # SKIPPED FOR THE SAE BRANCH, which captures its own per-chunk. These are FULL-BATCH forwards
    # storing 32 layers x [B, P, d] twice, so at the thousands of examples a compute-matched IxG
    # budget needs they OOM (measured: 11.5 GiB single alloc at B~7800 on an 80 GiB card) -- and
    # the SAE path never reads them, so it was paying for activations it discards.
    pt = tok(co, return_tensors="pt", padding=True).to(device)
    clean_acts, patch_acts = {}, {}
    if not hooker.is_sae:
        with torch.no_grad():
            clean_acts, _ = capture(bid, bam, False)
            patch_acts, _ = capture(pt.input_ids, pt.attention_mask, False)
        clean_acts = {k: v.detach() for k, v in clean_acts.items()}
        patch_acts = {k: v.detach() for k, v in patch_acts.items()}
    if hooker.zero_ablation and not hooker.is_sae:
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
    # ec/ep are FULL-BATCH embedding forwards; the SAE branch computes them per chunk instead,
    # for the same OOM reason as the captures above.
    if (ig_steps > 1 or conductance or hooker.include_input or mc) and not hooker.is_sae:
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

    if hooker.is_sae:
        # ---- IxG / IG over an SAE basis -------------------------------------------------
        # THE SAME ESTIMATOR AS THE MLP BRANCH BELOW, in the SAE's coordinates. There the score
        # is  g . (clean - patch)  with g the gradient at the node module's input; here the
        # "node" is a latent coefficient, so the score is  dmetric/df_j * (f_clean - f_patch)_j.
        #
        # dmetric/df_j comes from the chain rule through the decoder, which is LINEAR:
        # sae_loader.decode_delta(g) = (g @ W_dec.T) / s, so d(resid)/df_j = W_dec[:, j] / s and
        #     dmetric/df_j = (g_resid @ W_dec)_j / s.
        # That is exact, not an approximation -- the mask enters _sae_interchange linearly, which
        # is also why MAttr can train through it.
        #
        # The gradient is taken at the LAYER OUTPUT, because that is the tensor
        # _sae_interchange rewrites. ERROR NODE (index d_sae of each span block) gets the same
        # grad-dot-delta rule on the part of the residual the dictionary does not span:
        #     g . [(b - c) - decode_delta(f_b - f_c)].
        #
        # SIGN: hooker.sufficient is False for --mode sufficient, so the hook's live branch is
        # new = b + decode_delta((1-m_f)(f_c - f_b)) + (1-m_e) err_diff -- m=1 KEEPS CLEAN, the
        # repo-wide `sufficient` = denoising convention. Hence the delta is (clean - patch),
        # same orientation as the MLP branch's (c - p). Do not "fix" it to (patch - clean).
        #
        # CHUNKED over examples (--grad-batch). The score sums over the batch, so processing
        # examples in chunks and adding the partial sums is EXACT, not an approximation -- and
        # it is what makes a compute-matched budget possible at all: matching MAttr's 2000
        # backward passes means thousands of examples for IxG, and the single-batch version
        # stored 32 layers x [B, P, d_model] three times over (clean, patch, grad), which is
        # ~37 GB at B=4000. Chunking bounds that by the chunk size instead.
        mlp_site = hooker.mask_type == "mlp_sae_span"
        S_, W_, dsae = hooker.num_spans, hooker.sae_width, hooker.d_sae
        n_draws_ = ig_steps if (mc or ig_steps > 1) else 1
        chunk = grad_batch or len(cl)
        scores = torch.zeros(total)
        saved_mask, hooker.mask = hooker.mask, None   # interchange hooks no-op on mask=None

        def capture_sae(ids, am, want_grad, embed_override=None):
            store, handles = {}, []

            def mk(li):
                def hook(mod, inp, output):
                    x = output[0] if isinstance(output, tuple) else output
                    if want_grad:
                        # Model is frozen, so the layer output is effectively a leaf; make it one
                        # that requires grad (as capture() does for MLP) and RETURN it so
                        # everything downstream depends on the tensor whose .grad we read.
                        if not x.requires_grad:
                            x.requires_grad_(True)
                        x.retain_grad()
                        store[li] = x
                        return ((x,) + tuple(output[1:])) if isinstance(output, tuple) else x
                    store[li] = x
                    return output
                return hook
            for li_ in range(len(layers)):
                # SAME module resolution llama.py's register_hooks uses for the interchange, via
                # the hooker's own accessor rather than a hand-written path: for mlp_sae_span the
                # site is layer.mlp.down_proj, NOT layer.mlp. Their outputs happen to be the same
                # tensor, but capturing the gradient anywhere other than the exact tensor the
                # mask rewrites is the kind of near-miss that silently scores a different
                # intervention.
                mod_ = hooker._get_mlp_module(layers[li_]) if mlp_site else layers[li_]
                handles.append(mod_.register_forward_hook(mk(li_)))
            if embed_override is not None:
                handles.append(hf.model.embed_tokens.register_forward_hook(
                    lambda mod, inp, out: embed_override))
            logits = hf(ids, attention_mask=am).logits.float()
            for h_ in handles:
                h_.remove()
            return store, logits

        # attribution_loss reduces over the batch by MEAN, so a chunk's gradients carry a
        # 1/chunk factor where the single-batch run carries 1/N. Re-weighting each chunk by its
        # share of the total restores exactly the single-batch quantity -- without it the scores
        # come out (N/chunk) times too large (measured: 4x at N=100, chunk=25). The RANKING is
        # unaffected by a uniform scale, so this is invisible in acc_auc; it matters because the
        # scores are also written to .scores.pt and compared across runs.
        n_total = len(cl)
        n_used = 0
        for c0 in range(0, len(cl), chunk):
            kcl, kco = cl[c0:c0 + chunk], co[c0:c0 + chunk]
            kci, kii = ci[c0:c0 + chunk], ii[c0:c0 + chunk]
            kb = len(kcl); n_used += kb
            kt = tok(kcl, return_tensors="pt", padding=True).to(device)
            kid, kam = kt.input_ids, kt.attention_mask
            klast = kam.sum(1) - 1
            kpt = tok(kco, return_tensors="pt", padding=True).to(device)
            kcor = torch.tensor(kci, device=device); kinc = torch.tensor(kii, device=device)

            def kmetric(logits):
                ll = logits[torch.arange(kb, device=device), klast]
                return -attribution_loss(loss, ll, kcor, kinc, corrupt_topk=False,
                                         hinge_margin=hinge_margin, acc_temp=acc_temp,
                                         ld_scale=ld_scale)

            with torch.no_grad():
                kclean, _ = capture_sae(kid, kam, False)
                kpatch, _ = capture_sae(kpt.input_ids, kpt.attention_mask, False)
            kclean = {k: v.detach() for k, v in kclean.items()}
            kpatch = {k: v.detach() for k, v in kpatch.items()}

            kec = kep = None
            # `hooker.include_input` added 2026-08-30: the input node's score is
            # grad(emb).(clean_emb - patch_emb), so it needs both embeddings even at ig_steps=1
            # (I x G), where there is otherwise no interpolation and hence no reason to have them.
            if n_draws_ > 1 or mc or hooker.include_input:
                cap2 = {}
                h2 = hf.model.embed_tokens.register_forward_hook(
                    lambda m, i, o: cap2.__setitem__("e", o.detach()))
                with torch.no_grad():
                    hf(kid, attention_mask=kam); kec = cap2["e"]
                    hf(kpt.input_ids, attention_mask=kpt.attention_mask); kep = cap2["e"]
                h2.remove()

            kg = {li_: torch.zeros_like(v, dtype=torch.float32) for li_, v in kclean.items()}
            g_emb = torch.zeros_like(kec, dtype=torch.float32) if hooker.include_input else None
            for step in range(n_draws_):
                eo = None
                if mc:
                    a_ = torch.rand(kec.shape[0], 1, 1, generator=gen).to(kec)
                    eo = (1 - a_) * kec + a_ * kep
                elif n_draws_ > 1:
                    eo = (1 - step / n_draws_) * kec + (step / n_draws_) * kep
                elif hooker.include_input:
                    # ig_steps=1 and no MC: there is no interpolation, so the override is just the
                    # CLEAN embedding. It still has to be installed as a leaf, because the input
                    # node's score is the gradient AT that tensor and there is no other handle on
                    # it. Scoring the same point the forward would have used anyway means the
                    # layer scores are bit-identical to the -input run; only index 0 is new.
                    eo = kec.clone()
                if g_emb is not None:
                    eo = eo.detach().requires_grad_(True)
                store_g, logits = capture_sae(kid, kam, True, embed_override=eo)
                kmetric(logits).backward()
                for li_ in kg:
                    kg[li_] += store_g[li_].grad.float()
                if g_emb is not None and eo.grad is not None:
                    g_emb += eo.grad.float()
                del store_g

            bi_ = torch.tensor([SPAN_LAST[c] for c in kcl], device=device)
            si_ = torch.tensor([SPAN_LAST[c] for c in kco], device=device)
            for li_ in range(len(layers)):
                sae = hooker.saes[li_]
                dm_ = kclean[li_].shape[-1]
                bidx_ = bi_[:, :, None].expand(-1, -1, dm_)
                sidx_ = si_[:, :, None].expand(-1, -1, dm_)
                g_ = (kg[li_] / n_draws_).gather(1, bidx_).float()
                b_ = kclean[li_].gather(1, bidx_).float()
                c_ = kpatch[li_].gather(1, sidx_).float()
                fb_, fc_ = sae.encode(b_), sae.encode(c_)
                fd_ = (fb_ - fc_).float()
                gw_ = (g_ @ sae.W_dec.float()) / sae.s
                feat_ = (gw_ * fd_).sum(0)
                err_ = (b_ - c_) - sae.decode_delta(fd_).float()
                errs_ = (g_ * err_).sum(-1).sum(0)
                # No error column when the node is not part of the substrate -- W_ is d_sae
                # there, and concatenating it anyway would fail the shape assert below rather
                # than silently shifting every layer block by one, but be explicit.
                blk = (feat_ if hooker.sae_error == "none"
                       else torch.cat([feat_, errs_[:, None]], dim=-1))
                assert blk.shape == (S_, W_), f"SAE block {tuple(blk.shape)} != {(S_, W_)}"
                # SAME offset convention as llama.set_saes / _sae_interchange: index 0 is the
                # input-embedding node when +input, so every layer block shifts by one.
                off_ = hooker._node_offset + li_ * S_ * W_
                scores[off_:off_ + S_ * W_] += blk.reshape(-1).cpu() * (kb / n_total)
                del g_, b_, c_, fb_, fc_, fd_, gw_, feat_, err_, errs_, blk
            if g_emb is not None:
                # grad(emb).(clean - patch), averaged over the path exactly as the layer scores
                # are (kg/n_draws_), then chunk-weighted by kb/n_total like every other block.
                scores[0] += float(((g_emb / n_draws_) * (kec - kep).float()).sum()) * (kb / n_total)
                del g_emb
            del kclean, kpatch, kg
            torch.cuda.empty_cache()

        hooker.mask = saved_mask
        logger.info("SAE gradient attribution: %d examples x %d draws = %d example-backwards "
                    "(chunk=%d)", n_used, n_draws_, n_used * n_draws_, chunk)
        if modified_bwd:
            revert_relp(hf)
        return scores.to(device)

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


# The --adam-eps default, named so run_tag's "is this non-default?" test cannot drift from the
# argparse default the way a second literal would.
ADAM_EPS_DEFAULT = 1e-8


def eps_tag(x):
    """`1e-2` for 0.01 -- normalised scientific notation, one spelling per value.

    NOT f"{x:g}", which is inconsistent across the range this knob is swept over: it gives
    "0.01" for 1e-2 but "1e-08" for 1e-8, so the same sweep would spell two of its own points
    in two different notations. Normalising through f"{x:e}" makes the tag fragment identical
    to the directory names the eps grid already uses (results/adamsgd_mlp/A_eps/eps_1e-2_lr_*),
    so a run can be matched across the two layouts by eye.

    A shell submitter that must predict this filename passes the eps as the string "1e-2" and
    splices it in directly; that is exact for any value written in this normalised form, which
    is the only form worth passing.
    """
    mant, exp = f"{x:e}".split("e")
    return f"{float(mant):g}e{int(exp)}"


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
    # Adam eps. It sits HERE, next to the optimizer name it qualifies and ahead of the loss, so a
    # tag reads `sufficient_topk_adam_eps1e-2_ce_zeroabl_bs1` -- optimizer, then objective, then
    # setting. Written only when it is non-default, so every tag already on disk is unchanged.
    #
    # *** THIS IS AN IDENTITY KNOB, NOT A NUMERICAL ONE, AND OMITTING IT LOSES RUNS. ***
    # At neuron scale eps decides the ranking outright (see scripts/sva/launch/submit_adam_eps_followup.sh
    # for the measurement: acc-AUC 0.388 -> 0.490 on addition/mlp, IG-overlap 0.08 -> 0.73), so
    # two runs differing only in eps are two different circuits. Until 2026-08-28 the tag did not
    # carry it, which is why every eps run so far had to be quarantined in its own --output dir
    # (results/adamsgd_mlp/A_eps/eps_<e>_lr_<lr>/) -- inside one dir an eps run would have
    # silently OVERWRITTEN the default-eps run of the same cell. Encoding it is what lets the
    # eps arm live in results/sva_sweep beside the arm it is a one-knob change from.
    #
    # Only for mattr+adam: --adam-eps is passed to the optimizer only on that path (trainer.py's
    # kwargs branch), so appending it to an SGD or a gradient-method tag would claim a
    # distinction the run does not have.
    if args.method == "mattr" and args.optimizer == "adam" and args.adam_eps != ADAM_EPS_DEFAULT:
        tag += f"_eps{eps_tag(args.adam_eps)}"
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
    # `logit` needs its own fragment for the same reason every other non-default schedule has
    # one: without it a logit run writes the SAME filename as the `log` default and silently
    # overwrites it. Spelled `_logitk` to match the `_uniformk` / `_adaptivek` convention rather
    # than `_logit`, which reads as a loss name next to `logit_diff`.
    if args.method == "mattr" and args.k_schedule == "logit":
        tag += "_logitk"
    # Per-step grad normalisation changes the learned circuit, so it is part of the identity.
    # `:g` keeps 1.0 -> "gn1" and 0.01 -> "gn0.01" without a trailing ".0".
    # Substrate/intervention identity, not a method knob. "none" changes `total` and "frozen"
    # changes what every score MEANS, so neither may share a filename with an absorb run.
    # absorb stays unmarked so the existing results/sva_sweep SAE files keep their names.
    if args.sae_error == "none":
        tag += "_noerr"
    elif args.sae_error == "frozen":
        tag += "_ferr"
    if args.method == "mattr" and args.grad_norm > 0:
        tag += f"_gc{args.grad_norm:g}"
    if args.method == "mattr" and args.train_batch_size != 8:
        tag += f"_bs{args.train_batch_size}"
    if args.method in ("mattr", "edge_pruning", "sigmoid_mask") and args.steps != 2000:
        # edge_pruning/sigmoid_mask added 2026-09-06 (the NP/DBM 5k bump): without the suffix a
        # 5k run writes eprun_s090.json THROUGH the 5k tree's symlink and destroys the 2k
        # original. Consumers strip a trailing _s\d{4,} for these tags (parse_method), which
        # cannot collide with eprun's _s090 sparsity or mc_ig's _s42 seed.
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
    p.add_argument("--grad-batch", type=int, default=0, metavar="B",
                   help="physical minibatch for gradient attribution (0 = one batch, the old "
                        "behaviour). Scores sum over examples, so chunking is EXACT -- this only "
                        "bounds memory, and is what makes a large --grad-examples possible. "
                        "Currently honoured by the SAE branch only.")
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
    # topk_detached exposed 2026-08-29 for the residual-SAE diagnosis: it is `topk`'s forward with
    # tau DETACHED in the backward, i.e. it drops the -(sp_j/T)*gsp/sp_sum cross-score coupling
    # term that implicit differentiation of the budget constraint introduces. On resid_sae_span
    # that term is ~10^3x the discriminative signal and is a POSITIONAL confound (it scales with
    # sp_j = m_j(1-m_j), so with a unit's distance from tau, not its importance) -- and because it
    # scales with lr exactly as the signal does, no learning rate can fix the ratio. build_mask
    # has supported it since the start; only this choices list was gating it.
    p.add_argument("--variant", default="hard_topk",
                   choices=["topk", "topk_detached", "hard_topk",
                            "hard_topk_identity"])  # build_mask gate
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
    p.add_argument("--ld-scale", type=float, default=2.0, help="margin scale (logits) for --loss ld_tanh")
    p.add_argument("--adam-eps", type=float, default=1e-8,
                   help="Adam eps. At neuron scale this is a real knob, not a numerical guard: "
                        "below the typical |grad| Adam's update is ~sign(g) and all effect "
                        "magnitude is divided out; above it, Adam -> SGD+momentum at lr/eps.")
    p.add_argument("--adam-beta2", type=float, default=0.999, help="Adam beta2 for the scores")
    p.add_argument("--grad-norm", type=float, default=0.0,
                   help="MAttr: CLIP each step's score gradient to at most this norm (0 = off). "
                        "Scale-down only. Stops one large-k draw from setting the whole score "
                        "vector on the SAE basis, where |grad| spans ~10 orders of magnitude. "
                        "Not normalisation: scaling small gradients UP to the cap amplifies "
                        "numerically-dead steps and blows the scores to 1e21 (tried, far worse).")
    p.add_argument("--sgd-momentum", type=float, default=0.0,
                   help="SGD momentum for the scores. With --sgd-dampening equal to it, the "
                        "buffer is an EMA -- exactly the m-hat that big-eps Adam scales, minus "
                        "the per-coordinate 1/sqrt(v) normalisation.")
    p.add_argument("--sgd-dampening", type=float, default=0.0,
                   help="SGD dampening (0 = heavy-ball sum, =momentum -> EMA)")
    p.add_argument("--log-grad-stats", type=int, default=0, metavar="N",
                   help="every N steps, log percentiles of |dLoss/dscores|. This is the number "
                        "Adam's eps has to be compared against: below the typical |g| the "
                        "update is ~sign(g) and effect magnitude is discarded.")
    p.add_argument("--dump-per-example", action="store_true",
                   help="also store the per-example base-source margin at every sparsity grid "
                        "point (iso_metrics/cause_metrics key ld_per_example). Needed to tell a "
                        "high MEAN margin apart from a high FRACTION of decided examples.")
    p.add_argument("--sae-error", default="absorb", choices=["absorb", "frozen", "none"],
                   help="how the SAE reconstruction error is intervened on. absorb (legacy "
                        "default) measures the clean side against the BLENDED reconstruction, "
                        "which makes keeping one error node restore its whole (layer, position) "
                        "site regardless of the latents -- a master switch, and why MAttr puts "
                        "90-100%% of its top 10 on error nodes. frozen measures each side "
                        "against its own reconstruction, so the error node carries only the "
                        "unexplained residual and is an ordinary scored unit; exact at both "
                        "endpoints either way. none drops the node entirely (== frozen with "
                        "ke pinned to 1), which changes F_patch and so makes AUCs incomparable "
                        "to the other two.")
    p.add_argument("--sae-no-error", action="store_true",
                   help="deprecated alias for --sae-error none")
    p.add_argument("--sae-repo", default=None, help="Llama-Scope SAE repo (auto: LXM for mlp_sae_span, LXR for resid_sae_span)")
    p.add_argument("--sae-dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--das-dim", type=int, default=None, help="DAS rotation subspace rank (default d_model)")
    p.add_argument("--das-lr", type=float, default=1e-3, help="lr for the DAS rotation params")
    p.add_argument("--das-optimizer", default="adam", choices=["adam", "sgd"], help="optimizer for the DAS rotation (separate from --optimizer for scores)")
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--cf-cache-gb", type=float, default=4.0,
                   help="device-memory budget for the per-example CF-activation cache "
                        "(train steps AND the sparsity sweep reuse it); 0 disables")
    # `logit` exposed 2026-08-29 for the residual-SAE diagnosis; sample_k has implemented it all
    # along, only this choices list gated it. It samples alpha = k/total logit-uniformly, which
    # cancels sigmoid_topk's gate slope sp = alpha*(1-alpha) exactly, so a zero-init SGD run's
    # expected score IS activation-path IG (see schedules.sample_k for the derivation). That is
    # the direct antidote to what the SAE trace shows: under `log`, |grad| swings ~10 orders of
    # magnitude with k and one large-k draw sets the whole score vector (|s|max frozen from
    # step 3). Flattening the weight removes that lottery, and its fixed point is the estimator
    # that actually works on this basis.
    p.add_argument("--k-schedule", default="log",
                   choices=["uniform", "log", "logit", "adaptive_log", "log_both"])
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
    p.add_argument("--scores-from", nargs="+", default=None, metavar="LABEL:PATH",
                   help="skip training and run Phase-2 eval on each saved score vector instead "
                        "(cross-task transfer). PATH is a .pt holding either a raw [total] "
                        "tensor (eval_sva .scores.pt) or an eval_mib scores dict (its 'scores' "
                        "entry is used; same node layout, index 0 = input). Writes one json per "
                        "source with tag xfer_LABEL; sources whose json already exists are "
                        "skipped, so resubmission resumes.")
    p.add_argument("--output", default="results/sva")
    wandb_util.add_args(p)     # --no-wandb / --wandb-project / --wandb-entity; ON by default
    args = p.parse_args()
    if getattr(args, "sae_no_error", False):
        args.sae_error = "none"

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
        # arithmetic-wild has no CONTENT-span schema, so the content-span substrates would
        # silently fall back to a wrong NUM_SPANS; refuse them explicitly. The *_sae_span
        # substrates are NOT refused any more: they fall back to per-position spans below,
        # which need no schema (see SpanLast.set_per_position).
        assert args.nodes not in ("mlp_span", "mlp+attn_span", "mlp+attn_head_span",
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
        # The MIB fork is found the way every scripts/mib runner finds it (deps/MIB-circuit-track
        # first, then the legacy ./MIB-circuit-track symlink), not by a CWD-relative name.
        from learning_to_attribute.deps import add_mib_to_sys_path
        add_mib_to_sys_path()
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
    # PER-POSITION SAE. Only CGDataset builds a content-span schema, so on sva/arith/mib
    # NUM_SPANS is None and *_sae_span used to die at llama.py's "requires num_spans". Rather
    # than refuse those datasets, fall back to one span per TOKEN POSITION, which is exactly
    # what makes the SAE substrate behave "like the MLP neuron basis": `mlp` is one score per
    # (layer, position, neuron) and this is one per (layer, position, latent).
    #
    # VARLEN GOES BACK OFF when it does. Span mode normally skips the modal-length filter
    # because content spans normalise away length differences; per-position spans do not -- the
    # index list is 0..seq_len-1 and every pair in a batch must therefore be exactly seq_len
    # tokens, the same requirement `mlp` has.
    n_spans = NUM_SPANS
    if SAE and n_spans is None:
        n_spans = seq_len
        VARLEN = False
        SPAN_LAST.set_per_position(seq_len)
        logger.info("Per-position SAE substrate: %d spans = %d token positions (fixed length)",
                    n_spans, seq_len)
    corrupt_topk = args.mode == "necessary"
    hooker = LlamaAttributionHooks(hf, args.nodes, seq_len=seq_len,
                                   sufficient=corrupt_topk, include_input=args.include_input,
                                   num_spans=(n_spans if SPAN else None),
                                   zero_ablation=args.ablation == "zero",
                                   sae_error=args.sae_error)
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
        logger.info("Span mode: %d spans, %s", n_spans,
                    "variable length (no seq_len filter)" if VARLEN else f"fixed len={seq_len}")
    hooker.register_hooks()

    # --- per-string tokenization cache + per-example CF-activation cache -----------------
    # Every step (and every sweep cell) used to re-tokenize its strings (3 tokenizer calls
    # per accepted example) and re-run the CF forward for examples whose activations are
    # deterministic. tok_batch reproduces tokenizer(strings, padding=True) bit-for-bit from
    # cached per-string ids (right padding); CFActivationCache runs the CF forward only on
    # a batch's uncached examples (keyed by the corrupted string) -- in the sparsity sweep,
    # where the same eval batches recur at ~24 sparsities per direction, the CF forwards
    # collapse to the first pass.
    _ids_cache = {}

    def _ids(s):
        v = _ids_cache.get(s)
        if v is None:
            v = tok(s, return_tensors="pt").input_ids[0]
            _ids_cache[s] = v
        return v

    def tok_batch(strings):
        seqs = [_ids(s) for s in strings]
        lens = torch.tensor([x.shape[0] for x in seqs])
        P = int(lens.max())
        ids = torch.full((len(seqs), P), tok.pad_token_id, dtype=torch.long)
        for b, x in enumerate(seqs):
            ids[b, : x.shape[0]] = x
        attn = (torch.arange(P)[None] < lens[:, None]).long()
        return ids.to(device), attn.to(device), lens

    cf_cache = CFActivationCache(hooker, max_gb=args.cf_cache_gb, logger=logger)

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
                if _ids(clean).shape[0] != seq_len or _ids(corr).shape[0] != seq_len:
                    continue
            elif not SPAN:  # node: interpolation needs clean/corrupted the SAME length per pair
                if _ids(clean).shape[0] != _ids(corr).shape[0]:
                    continue
            cl.append(clean); co.append(corr); ci.append(lab[0]); ii.append(lab[1])
        return cl, co, ci, ii

    def forward_last(cleans, corrupteds, ci, ii, mask, sufficient):
        """Run the masked forward; return (last-token logits [B,vocab], base idx, source idx)."""
        b_ids, b_attn, b_lens = tok_batch(cleans)
        s_ids, _, s_lens = tok_batch(corrupteds)
        last = (b_lens - 1).to(device)
        cf_cache.prepare(corrupteds, s_ids, s_lens.tolist())
        set_span(cleans, corrupteds)
        old_suf = hooker.sufficient; hooker.sufficient = sufficient
        hooker.mask = mask
        # model dtype until after the last-token gather; .float() on the full [B,P,vocab]
        # tensor put ~100 MB fp32 in the graph for a gradient nonzero at one position.
        logits = hf(b_ids, attention_mask=b_attn).logits
        hooker.sufficient = old_suf
        B = len(cleans)
        ll = logits[torch.arange(B, device=device), last].float()
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

    # Per-example CLEAN margins for the ld_match* losses, computed lazily and cached by clean
    # text: mask=ones + sufficient=False is exactly the F_clean forward summarize() uses, so
    # the target is the same quantity the faithfulness eval calls "the full model". Each of
    # the ~1.3k train examples pays one extra no-grad forward ONCE across the whole run.
    _clean_d = {}

    def clean_margins(cl, co, ci, ii):
        miss = [i for i, c in enumerate(cl) if c not in _clean_d]
        if miss:
            with torch.no_grad():
                ll, cor, inc = forward_last([cl[i] for i in miss], [co[i] for i in miss],
                                            [ci[i] for i in miss], [ii[i] for i in miss],
                                            torch.ones(total, device=device), sufficient=False)
                ar = torch.arange(ll.shape[0], device=device)
                dv = ll[ar, cor] - ll[ar, inc]
            for j, i in enumerate(miss):
                _clean_d[cl[i]] = float(dv[j])
        return torch.tensor([_clean_d[c] for c in cl], device=device)

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
            tgt = clean_margins(cl, co, ci, ii) if args.loss in CLEAN_TARGET_LOSSES else None
            return attribution_loss(args.loss, ll, cor, inc, corrupt_topk=step_cause,
                                    hinge_margin=args.hinge_margin, acc_temp=args.acc_temp,
                                    ld_scale=args.ld_scale, target_d=tgt)
        # ---- MAttr-IG: integrate dL/dmask over the baseline(CF)->clean mask path ----
        # effective mask alpha*m_hard makes activations cf + alpha*m*(clean-cf): alpha=0 is the
        # all-baseline circuit, alpha=1 the top-k intervention. a_ig_j = mean_alpha dL/d(mask_j)
        # is the IG attribution of node j; the surrogate (a_ig * mask).sum() re-routes it to the
        # scores through mask's STE (identity or sigmoid), so no trainer change is needed.
        b_ids, b_attn, b_lens = tok_batch(cl)
        s_ids, _, s_lens = tok_batch(co)
        last = (b_lens - 1).to(device)
        cf_cache.prepare(co, s_ids, s_lens.tolist())
        set_span(cl, co)
        B = len(cl); ar = torch.arange(B, device=device)
        cor = torch.tensor(ci, device=device); inc = torch.tensor(ii, device=device)
        m_hard = mask.detach()
        a_ig = torch.zeros_like(m_hard); L1 = None
        old_suf = hooker.sufficient; hooker.sufficient = step_cause
        for j in range(1, IG_STEPS + 1):
            mm = (float(j) / IG_STEPS * m_hard).requires_grad_(True)
            hooker.mask = mm
            logits = hf(b_ids, attention_mask=b_attn).logits
            Lj = attribution_loss(args.loss, logits[ar, last].float(), cor, inc, corrupt_topk=step_cause,
                                  hinge_margin=args.hinge_margin, acc_temp=args.acc_temp,
                                  ld_scale=args.ld_scale)
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
            b_ids, b_attn, b_lens = tok_batch(xc[s:s+20])
            s_ids, _, s_lens = tok_batch(xco[s:s+20])
            last = (b_lens - 1).to(device)
            cf_cache.prepare(xco[s:s+20], s_ids, s_lens.tolist())
            set_span(xc[s:s+20], xco[s:s+20])
            old = hooker.sufficient; hooker.sufficient = sufficient; hooker.mask = mask.to(device)
            logits = hf(b_ids, attention_mask=b_attn).logits
            hooker.sufficient = old
            B = b_ids.shape[0]; ar = torch.arange(B, device=device)
            ll = logits[ar, last].float(); probs = ll.softmax(-1)
            cor = torch.tensor(xci[s:s+20], device=device); inc = torch.tensor(xii[s:s+20], device=device)
            LB.append(ll[ar, cor]); LS.append(ll[ar, inc]); PB.append(probs[ar, cor]); PS.append(probs[ar, inc])
        lb = torch.cat(LB); ls = torch.cat(LS); pb = torch.cat(PB); ps = torch.cat(PS)
        ld = lb - ls
        return {**({"ld_per_example": ld.tolist()} if args.dump_per_example else {}),
                "logit_diff": ld.mean().item(),
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
            if args.log_grad_stats and step % args.log_grad_stats == 0 \
                    and live_scores.grad is not None:
                # learn_scores calls on_step AFTER optimizer_.step() but BEFORE the next
                # iteration's zero_grad(), so .grad still holds this step's gradient.
                g = live_scores.grad.detach().abs().float()
                nz = g[g > 0]
                qs = torch.tensor([0.5, 0.9, 0.99, 1.0], device=g.device)
                q = torch.quantile(nz[::max(1, nz.numel() // 200000)], qs) if nz.numel() \
                    else torch.zeros(4)
                logger.info("  [grad %5d] nonzero=%.4f  |g| p50=%.3e p90=%.3e p99=%.3e "
                            "max=%.3e  (adam eps=%.1e)", step,
                            float((g > 0).float().mean()), *[float(x) for x in q],
                            args.adam_eps)
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
    if args.scores_from:
        scores = None  # per-source vectors are loaded in the eval loop below; nothing trains
    elif args.method == "random":
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
                                 ld_scale=args.ld_scale,
                                 conductance=cond, grad_batch=args.grad_batch,
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
                           lr=args.lr, optimizer=args.optimizer, device=device,
                           adam_eps=args.adam_eps, adam_betas=(0.9, args.adam_beta2),
                           sgd_momentum=args.sgd_momentum, sgd_dampening=args.sgd_dampening,
                           grad_norm=args.grad_norm,
                           k_sampler=k_sampler, extra_params=das_params, lr_extra=args.das_lr,
                           extra_optimizer=(args.das_optimizer if DAS else None),
                           on_step=on_step_cb,
                           # This call site was the only learn_scores/edge_pruning/sigmoid_mask
                           # one without a logger, so MAttr runs printed no per-step trace at all
                           # -- which is why the residual-SAE score runaway (|s| reaching 1e10)
                           # went unnoticed through a whole sweep. MATTR_LOG_EVERY=1 gives the
                           # per-step |grad| needed to catch it near init, where it is worst.
                           logger=logger,
                           log_every=int(os.environ.get("MATTR_LOG_EVERY", "200")))
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

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    if args.scores_from:
        # Cross-task transfer: evaluate someone ELSE's ranking on this task's eval set. One
        # json per source (tag xfer_LABEL); cells whose json already exists are skipped, so a
        # resubmitted job resumes instead of recomputing. The model load + eval-set collection
        # above amortize over all sources, which is why this is a list and not one job each.
        runs = []
        for spec in args.scores_from:
            label, _, pth = spec.partition(":")
            if not pth:
                raise SystemExit(f"--scores-from wants LABEL:PATH, got {spec!r}")
            sd = torch.load(pth, map_location="cpu")
            vec = (sd["scores"] if isinstance(sd, dict) else sd).float().flatten()
            if vec.numel() != total:
                raise SystemExit(f"{pth}: {vec.numel()} scores but this substrate has {total} "
                                 "units -- source and target node layouts do not match")
            runs.append((f"xfer_{label}", vec, spec))
    else:
        runs = [(tag, scores, None)]

    for tag_i, scores_i, src_i in runs:
        fn = outdir / f"{args.task}_{args.model}_{args.nodes.replace('+','-')}_{tag_i}.json"
        if src_i is not None and fn.exists():
            logger.info("skip %s (already on disk)", fn)
            continue
        S = summarize(scores_i, (ec, eco, eci, eii))   # eval_metrics/summarize defined above (pre-training)
        FM, F0 = S["F_clean"], S["F_patch"]
        faith_auc, acc_auc, kstar_50 = S["faith_auc"], S["acc_auc"], S["kstar_50"]
        logger.info("[%s] F(clean)=%.3f  F(patch)=%.3f", tag_i, FM, F0)
        logger.info("[%s] acc_auc=%.3f  k*(>0.5)=%s  k*(>0.9)=%s", tag_i, acc_auc, kstar_50, S["kstar_90"])

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
        if src_i is not None:
            out["xfer_source"] = src_i   # LABEL:PATH of the ranking actually evaluated
        torch.save(scores_i.cpu(), fn.with_suffix(".scores.pt"))
        json.dump(out, open(fn, "w"), indent=2)
        logger.info("iso/faith AUC=%.3f (fmax %.3f) | cause AUC=%.3f | total=%d -> %s",
                    S["faith_auc"], S["faith_max"], S["cause_auc"], total, fn)
    logger.info(cf_cache.stats())
    hooker.remove_hooks()

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

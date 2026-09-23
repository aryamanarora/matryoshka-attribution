"""Per-model driver for the cross-format arithmetic replication (arXiv:2609.04463) with MAttr.

Everything the paper does with attribution patching is done here twice: with the paper's own
attribution patching (AP, ``(a(x) - a(x')) * dM/da`` at x') and with MAttr (a learned soft
top-k mask over the same units, trained in the iso/denoising direction: top-k units keep
their clean-prompt activation, the rest take the sign-flipped prompt's activation, and the loss
pushes the model's preference back toward its clean answer).

Units: every MLP neuron (input of ``down_proj``) of every layer at the LAST PROMPT TOKEN.
Metric: m(z) = log P(y_hat | z) - log P(y_hat' | z), teacher-forced sum over answer tokens,
where y_hat / y_hat' are the model's own greedy answers to x / x'.  M(z) rescales m to [0, 1]
between m(x') and m(x).

Stages (``--stage all`` runs them in order; each stage is skipped if its outputs exist):
  behav   greedy answers, correctness, m(x), m(x'), mean log-prob, entropy   -> behav_<fmt>.json
  acts    clean / flipped activations at the last prompt token                -> acts_<fmt>.pt
  ap      format-level AP means (all / correct / incorrect items)             -> ap_<fmt>.pt
  mattr   format-level MAttr scores (+ correct/incorrect subsets, numeric)    -> mattr_<fmt>[_<sub>].pt
  circuits top-1% circuits, Jaccard, sufficiency curves (MAttr / AP / random) -> circuits.json
  items   per-item AP loadings on every circuit + patch-restored fractions    -> items_<fmt>.json
  itemmattr per-item MAttr loadings (subset of items; NOT in "all")           -> itemmattr_<fmt>.json

Outputs go to ``results/arith_formats/<model>/``.
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FORMATS, gold, is_correct, make_items, prompt  # noqa: E402

from matryoshka_attribution.sigmoid_topk import sigmoid_topk  # noqa: E402
from matryoshka_attribution.trainer import learn_scores  # noqa: E402

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",
    "qwen3-4b": "Qwen/Qwen3-4B-Base",
    "qwen3-8b": "Qwen/Qwen3-8B-Base",
    "qwen3-32b": "Qwen/Qwen3-32B",            # no base release of the 32B
    "llama3.2-1b": "meta-llama/Llama-3.2-1B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "olmo3-7b": "allenai/Olmo-3-1025-7B",
    "olmo3-32b": "allenai/Olmo-3-1125-32B",
    "phi4-14b": "microsoft/phi-4",
}

log = logging.getLogger("arith_formats")


# ----------------------------------------------------------------------------- hooks
class LastTokenMLP:
    """Forward-pre-hooks on every layer's ``mlp.down_proj`` acting only at row-specific
    positions ``lp`` (the last prompt token).

    mode None      : pass-through
    mode "capture" : store the (rows, d) activation of every layer in ``self.acts``
    mode "grad"    : replace it by a leaf that requires grad (``self.leaves``)
    mode "blend"   : new = sel + m * (cf - sel), m = ``self.m[:, li]`` (rows or 1, d) in
                     [0, 1] = fraction taken from ``self.cf[:, li]`` (rows, d)
    """

    def __init__(self, hf):
        self.hf = hf
        self.layers = hf.model.layers
        self.L = len(self.layers)
        self.d = hf.config.intermediate_size
        self.mode = None
        self.ar = self.lp = None
        self.m = self.cf = None
        self.acts, self.leaves = {}, {}
        self._handles = [layer.mlp.down_proj.register_forward_pre_hook(self._hook(li))
                         for li, layer in enumerate(self.layers)]

    def _hook(self, li):
        def hook(mod, args):
            if self.mode is None:
                return None
            x = args[0]
            sel = x[self.ar, self.lp]                       # (rows, d)
            if self.mode == "capture":
                self.acts[li] = sel.detach()
                return None
            if self.mode == "grad":
                leaf = sel.detach().float().requires_grad_(True)
                self.leaves[li] = leaf
                new = leaf
            elif self.mode == "blend":
                m = self.m[:, li]
                cf = self.cf[:, li].float()
                s = sel.float()
                new = s + m * (cf - s)
            else:
                raise ValueError(self.mode)
            x2 = x.clone()
            x2[self.ar, self.lp] = new.to(x.dtype)
            return (x2,)
        return hook

    def set_rows(self, lp):
        self.lp = lp
        self.ar = torch.arange(lp.shape[0], device=lp.device)

    def remove(self):
        for h in self._handles:
            h.remove()


# ----------------------------------------------------------------------------- model glue
class Runner:
    def __init__(self, key, device, out):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = MODELS[key]
        self.key, self.device, self.out = key, device, out
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        attn = "eager" if "gemma" in name.lower() else "sdpa"
        t0 = time.time()
        self.hf = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, attn_implementation=attn).to(device).eval()
        for p in self.hf.parameters():
            p.requires_grad_(False)
        log.info("loaded %s in %.0fs (%s)", name, time.time() - t0, attn)
        self.hook = LastTokenMLP(self.hf)
        self.L, self.d = self.hook.L, self.hook.d
        self.total = self.L * self.d
        log.info("L=%d d_mlp=%d total=%d", self.L, self.d, self.total)
        self._pid = {}

    # -- tokenisation
    def pids(self, text):
        if text not in self._pid:
            self._pid[text] = self.tok(text, add_special_tokens=True)["input_ids"]
        return self._pid[text]

    def batch(self, prompts, answers):
        """Right-padded rows of prompt+answer; returns ids, attn, lp (last prompt idx), and
        (tgt_pos, tgt_ids, tgt_mask) for teacher forcing."""
        rows, lps, ans = [], [], []
        for p, a in zip(prompts, answers):
            pi = self.pids(p)
            rows.append(pi + list(a)); lps.append(len(pi) - 1); ans.append(list(a))
        T = max(len(r) for r in rows)
        A = max(1, max(len(a) for a in ans))
        ids = torch.full((len(rows), T), self.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(rows), T), dtype=torch.long)
        tpos = torch.zeros((len(rows), A), dtype=torch.long)
        tids = torch.zeros((len(rows), A), dtype=torch.long)
        tmask = torch.zeros((len(rows), A))
        for b, (r, lp, a) in enumerate(zip(rows, lps, ans)):
            ids[b, :len(r)] = torch.tensor(r); attn[b, :len(r)] = 1
            for t, tokid in enumerate(a):
                tpos[b, t] = lp + t; tids[b, t] = tokid; tmask[b, t] = 1.0
        dev = self.device
        return (ids.to(dev), attn.to(dev), torch.tensor(lps).to(dev),
                tpos.to(dev), tids.to(dev), tmask.to(dev))

    def tf_logprob(self, prompts, answers, need_last_logits=False):
        """Teacher-forced sum log P(answer | prompt) per row (float32, on the graph)."""
        ids, attn, lp, tpos, tids, tmask = self.batch(prompts, answers)
        self.hook.set_rows(lp)
        logits = self.hf(input_ids=ids, attention_mask=attn).logits
        B = ids.shape[0]
        ar = torch.arange(B, device=self.device)[:, None]
        sel = logits[ar, tpos].float()                                  # (B, A, V)
        lps = torch.log_softmax(sel, -1).gather(-1, tids[..., None])[..., 0]
        tot = (lps * tmask).sum(1)
        if need_last_logits:
            return tot, logits[torch.arange(B, device=self.device), lp].float()
        return tot

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=20):
        tok = self.tok
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(self.device)
        out = self.hf.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id, temperature=None, top_p=None,
                               top_k=None)
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        tok.padding_side = "right"
        # The answer token sequence INCLUDES its terminator (the newline / EOS token the model
        # produced): without it, an answer that is a token-prefix of the other (" 2" vs " 284")
        # would trivially have the higher summed log-prob.  ``text`` is the content only.
        res = []
        for g in gen:
            ans, text, done = [], "", False
            for t in g:
                if t == tok.eos_token_id or t == tok.pad_token_id:
                    if text.strip():
                        ans.append(t)
                    done = True
                    break
                piece = tok.decode(ans + [t])[len(text):]
                if "\n" in piece:
                    pre = piece.split("\n")[0]
                    text += pre
                    if text.strip():
                        ans.append(t)
                    done = True
                    break
                ans.append(t); text += piece
            if not text.strip():
                ans = []
            res.append((ans, text))
        return res


# ----------------------------------------------------------------------------- stages
def stage_behav(R, items, fmt, bs):
    fn = R.out / f"behav_{fmt}.json"
    if fn.exists():
        return json.load(open(fn))
    log.info("[behav %s] generating", fmt)
    P = [prompt(it, fmt) for it in items]
    Pf = [prompt(it, fmt, flipped=True) for it in items]
    gens, gensf = [], []
    for i in range(0, len(items), bs):
        gens += R.generate(P[i:i + bs]); gensf += R.generate(Pf[i:i + bs])
    rec = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            sl = slice(i, i + bs)
            ys = [g[0] for g in gens[sl]]; yfs = [g[0] for g in gensf[sl]]
            ys_ = [y if y else [R.tok.eos_token_id] for y in ys]
            yfs_ = [y if y else [R.tok.eos_token_id] for y in yfs]
            R.hook.mode = None
            lp_y_x, last = R.tf_logprob(P[sl], ys_, need_last_logits=True)
            lp_yf_x = R.tf_logprob(P[sl], yfs_)
            lp_y_xf = R.tf_logprob(Pf[sl], ys_)
            lp_yf_xf = R.tf_logprob(Pf[sl], yfs_)
            ent = -(torch.softmax(last, -1) * torch.log_softmax(last, -1)).sum(-1)
            for j, it in enumerate(items[sl]):
                y, yt = gens[i + j]; yf, yft = gensf[i + j]
                valid = bool(y) and bool(yf) and y != yf
                rec.append({
                    "id": it["id"], "gen": yt, "gen_ids": y, "gen_flip": yft, "gen_flip_ids": yf,
                    "correct": is_correct(yt, it, fmt),
                    "correct_flip": bool(yf) and is_correct(yft, {**it, "answer": it["answer_flip"]}, fmt),
                    "lp_y_x": float(lp_y_x[j]), "lp_yf_x": float(lp_yf_x[j]),
                    "lp_y_xf": float(lp_y_xf[j]), "lp_yf_xf": float(lp_yf_xf[j]),
                    "m_x": float(lp_y_x[j] - lp_yf_x[j]), "m_xf": float(lp_y_xf[j] - lp_yf_xf[j]),
                    "mean_logprob": float(lp_y_x[j]) / max(1, len(y)),
                    "entropy": float(ent[j]), "valid": valid,
                })
    acc = np.mean([r["correct"] for r in rec])
    nval = sum(r["valid"] for r in rec)
    log.info("[behav %s] acc=%.3f valid=%d/%d  example: %r -> %r (gold %r)", fmt, acc, nval,
             len(rec), P[0].split("\n")[-1], rec[0]["gen"], gold(items[0], fmt))
    json.dump({"format": fmt, "acc": float(acc), "n_valid": nval, "items": rec}, open(fn, "w"))
    return json.load(open(fn))


def stage_acts(R, items, fmt, bs):
    """Clean and flipped activations at the last prompt token, (N, L, d) bf16 on CPU."""
    fn = R.out / f"acts_{fmt}.pt"
    if fn.exists():
        return torch.load(fn)
    log.info("[acts %s] (recomputed each run; not persisted -- the shared disk is nearly full)", fmt)
    res = {}
    for flipped, name in ((False, "clean"), (True, "flip")):
        P = [prompt(it, fmt, flipped=flipped) for it in items]
        out = torch.empty((len(items), R.L, R.d), dtype=torch.bfloat16)
        with torch.no_grad():
            for i in range(0, len(items), bs):
                R.hook.mode = "capture"; R.hook.acts = {}
                R.tf_logprob(P[i:i + bs], [[R.tok.eos_token_id]] * len(P[i:i + bs]))
                out[i:i + bs] = torch.stack([R.hook.acts[l] for l in range(R.L)], 1).cpu()
        R.hook.mode = None
        res[name] = out
    if os.environ.get("ARITH_SAVE_ACTS"):
        torch.save(res, fn)
    return res


class LazyActs:
    """Load ``acts_<fmt>.pt`` on demand, keeping at most one format's cache in RAM."""

    def __init__(self, R, items, fmts, bs):
        self.R, self.items, self.fmts, self.bs = R, items, fmts, bs
        self._fmt, self._val = None, None

    def __getitem__(self, fmt):
        if fmt != self._fmt:
            self._val = None
            self._val = stage_acts(self.R, self.items, fmt, self.bs)
            self._fmt = fmt
        return self._val


MIN_MARGIN = 1.0   # nats; set from --min-margin


def _valid_ids(beh):
    """Items whose two greedy answers differ, are non-empty, and are separated by at least
    MIN_MARGIN nats of preference (m(x) - m(x')); tiny margins make M = (m - m(x'))/(m(x) - m(x'))
    explode."""
    return [r["id"] for r in beh["items"] if r["valid"] and (r["m_x"] - r["m_xf"]) >= MIN_MARGIN]


def ap_batch(R, items, beh, acts, fmt, idx, units=None):
    """Per-item AP scores for items ``idx`` (list of ids) -> (len(idx), L, d) float32 CPU,
    or restricted to flat unit indices ``units`` -> (len(idx), len(units))."""
    B = len(idx)
    recs = [beh["items"][i] for i in idx]
    Pf = [prompt(items[i], fmt, flipped=True) for i in idx]
    ys = [r["gen_ids"] for r in recs]; yfs = [r["gen_flip_ids"] for r in recs]
    R.hook.mode = "grad"; R.hook.leaves = {}
    lp = R.tf_logprob(Pf + Pf, ys + yfs)                   # rows: [x'+y | x'+y']
    m = lp[:B] - lp[B:]
    mx = torch.tensor([r["m_x"] for r in recs], device=R.device)
    mxf = torch.tensor([r["m_xf"] for r in recs], device=R.device)
    M = (m - mxf) / (mx - mxf)
    M.sum().backward()
    R.hook.mode = None
    g = torch.stack([R.hook.leaves[l].grad for l in range(R.L)], 1)   # (2B, L, d)
    g = g[:B] + g[B:]
    delta = (acts["clean"][idx].to(R.device).float() - acts["flip"][idx].to(R.device).float())
    s = (g * delta)                                                     # (B, L, d)
    R.hook.leaves = {}
    if units is not None:
        return s.reshape(B, -1)[:, units].cpu()
    return s.cpu()


def stage_ap(R, items, beh, acts, fmt, bs):
    fn = R.out / f"ap_{fmt}.pt"
    if fn.exists():
        return torch.load(fn)
    log.info("[ap %s]", fmt)
    ids = _valid_ids(beh)
    corr = {r["id"]: r["correct"] for r in beh["items"]}
    acc = {"all": torch.zeros(R.L, R.d, dtype=torch.float64), "correct": torch.zeros(R.L, R.d, dtype=torch.float64),
           "incorrect": torch.zeros(R.L, R.d, dtype=torch.float64)}
    n = {"all": 0, "correct": 0, "incorrect": 0}
    for i in range(0, len(ids), bs):
        idx = ids[i:i + bs]
        s = ap_batch(R, items, beh, acts, fmt, idx).double()
        acc["all"] += s.sum(0); n["all"] += len(idx)
        c = torch.tensor([corr[j] for j in idx])
        if c.any():
            acc["correct"] += s[c].sum(0); n["correct"] += int(c.sum())
        if (~c).any():
            acc["incorrect"] += s[~c].sum(0); n["incorrect"] += int((~c).sum())
    res = {k: (acc[k] / max(1, n[k])).float() for k in acc}
    res["n"] = n
    torch.save(res, fn)
    return res


def mattr_loss_factory(R, items, beh, acts, fmt, ids, bs, norm, rng):
    """loss_fn(mask) for learn_scores: mask (L*d,) with 1 = keep clean."""
    recs = {r["id"]: r for r in beh["items"]}

    def loss_fn(mask):
        idx = rng.sample(ids, bs)
        P = [prompt(items[i], fmt) for i in idx]
        ys = [recs[i]["gen_ids"] for i in idx]; yfs = [recs[i]["gen_flip_ids"] for i in idx]
        R.hook.mode = "blend"
        R.hook.m = (1.0 - mask).view(1, R.L, R.d)
        cf = acts["flip"][idx].to(R.device)
        R.hook.cf = torch.cat([cf, cf], 0)
        lp = R.tf_logprob(P + P, ys + yfs)
        m = lp[:bs] - lp[bs:]
        R.hook.mode = None
        if norm:
            mx = torch.tensor([recs[i]["m_x"] for i in idx], device=R.device)
            mxf = torch.tensor([recs[i]["m_xf"] for i in idx], device=R.device)
            return -((m - mxf) / (mx - mxf)).mean()
        return -m.mean()
    return loss_fn


def stage_mattr(R, items, beh, acts, fmt, args, subset=None):
    tag = f"mattr_{fmt}" + (f"_{subset}" if subset else "")
    fn = R.out / f"{tag}.pt"
    if fn.exists():
        return torch.load(fn)
    ids = _valid_ids(beh)
    if subset:
        corr = {r["id"]: r["correct"] for r in beh["items"]}
        ids = [i for i in ids if corr[i] == (subset == "correct")]
    if len(ids) < args.train_bs:
        log.warning("[%s] only %d valid items; skipping", tag, len(ids))
        return None
    log.info("[%s] training on %d items, %d steps, %s lr=%g norm=%s", tag, len(ids), args.steps,
             args.optimizer, args.lr, args.loss_norm)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    loss_fn = mattr_loss_factory(R, items, beh, acts, fmt, ids, args.train_bs, args.loss_norm, rng)
    res = learn_scores(R.total, loss_fn, steps=args.steps, variant=args.variant, k_schedule="log",
                       T=args.T, n_iters=30, lr=args.lr, optimizer=args.optimizer,
                       adam_eps=args.adam_eps, device=R.device, log_every=200, logger=log)
    out = {"scores": res.scores.view(R.L, R.d), "loss_log": res.loss_log, "k_log": res.k_log,
           "train_time_s": res.train_time_s, "n_items": len(ids),
           "config": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}}
    torch.save(out, fn)
    return out


# ----------------------------------------------------------------------------- circuits
def topk_units(scores, frac):
    """Flat indices of the top ``frac`` fraction of units by (signed) score."""
    k = max(1, int(round(frac * scores.numel())))
    return torch.topk(scores.flatten().float(), k).indices


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a | b)


@torch.no_grad()
def restored_fraction(R, items, beh, acts, fmt, idx, units, bs, cache=None):
    """Patch ``units`` (flat indices) at the last prompt token of the FLIPPED run with their
    clean values; return per-item M in [0,1] (restored fraction of the preference)."""
    recs = {r["id"]: r for r in beh["items"]}
    mvec = torch.zeros(R.total, device=R.device)
    mvec[units] = 1.0
    R.hook.m = mvec.view(1, R.L, R.d)
    out = []
    for i in range(0, len(idx), bs):
        sub = idx[i:i + bs]
        B = len(sub)
        Pf = [prompt(items[j], fmt, flipped=True) for j in sub]
        ys = [recs[j]["gen_ids"] for j in sub]; yfs = [recs[j]["gen_flip_ids"] for j in sub]
        R.hook.mode = "blend"
        cf = acts["clean"][sub].to(R.device)
        R.hook.cf = torch.cat([cf, cf], 0)
        lp = R.tf_logprob(Pf + Pf, ys + yfs)
        R.hook.mode = None
        m = lp[:B] - lp[B:]
        mx = torch.tensor([recs[j]["m_x"] for j in sub], device=R.device)
        mxf = torch.tensor([recs[j]["m_xf"] for j in sub], device=R.device)
        out.append(((m - mxf) / (mx - mxf)).cpu())
    return torch.cat(out)


def random_matched(units, L, d, seed):
    """Random units matched in number to ``units`` within each layer."""
    g = torch.Generator().manual_seed(seed)
    per_layer = torch.bincount(units // d, minlength=L)
    out = []
    for l in range(L):
        n = int(per_layer[l])
        if n:
            out.append(l * d + torch.randperm(d, generator=g)[:n])
    return torch.cat(out)


def stage_circuits(R, items, behs, actss, aps, mattrs, args):
    fn = R.out / "circuits.json"
    if fn.exists() and not args.force_circuits:
        return json.load(open(fn))
    log.info("[circuits]")
    frac = args.circuit_frac
    circ = {}
    for fmt in FORMATS:
        circ[("ap", fmt)] = topk_units(aps[fmt]["all"], frac)
        if mattrs.get(fmt) is not None:
            circ[("mattr", fmt)] = topk_units(mattrs[fmt]["scores"], frac)
    for sub in ("correct", "incorrect"):
        circ[("ap", f"numeric_{sub}")] = topk_units(aps["numeric"][sub], frac)
        if mattrs.get(f"numeric_{sub}") is not None:
            circ[("mattr", f"numeric_{sub}")] = topk_units(mattrs[f"numeric_{sub}"]["scores"], frac)
    out = {"frac": frac, "k": int(circ[("ap", "numeric")].numel()), "L": R.L, "d": R.d,
           "total": R.total, "jaccard": {}, "layer_hist": {}, "curves": {}, "acc": {},
           "n_valid": {}}
    for fmt in FORMATS:
        out["acc"][fmt] = behs[fmt]["acc"]; out["n_valid"][fmt] = len(_valid_ids(behs[fmt]))
    keys = list(circ)
    for a in keys:
        out["layer_hist"][f"{a[0]}:{a[1]}"] = torch.bincount(circ[a] // R.d, minlength=R.L).tolist()
        for b in keys:
            out["jaccard"][f"{a[0]}:{a[1]}|{b[0]}:{b[1]}"] = jaccard(circ[a], circ[b])
    # method agreement on the rankings (Spearman over the union of the two top-1% sets)
    # sufficiency curves: restored fraction vs #units, on n_eval valid items of each format
    rng = random.Random(args.seed)
    for fmt in FORMATS:
        ids = _valid_ids(behs[fmt])
        if not ids:
            continue
        ev = sorted(rng.sample(ids, min(args.n_eval, len(ids))))
        rankings = {"ap": aps[fmt]["all"].flatten(), "random": torch.randn(R.total, generator=torch.Generator().manual_seed(args.seed))}
        if mattrs.get(fmt) is not None:
            rankings["mattr"] = mattrs[fmt]["scores"].flatten()
        if fmt != "numeric":     # the numeric circuit applied to a verbal format
            rankings["ap_numeric"] = aps["numeric"]["all"].flatten()
            if mattrs.get("numeric") is not None:
                rankings["mattr_numeric"] = mattrs["numeric"]["scores"].flatten()
        curves = {}
        for name, sc in rankings.items():
            curves[name] = {}
            for f in args.curve_fracs:
                u = topk_units(sc, f)
                rf = restored_fraction(R, items, behs[fmt], actss[fmt], fmt, ev, u, args.bs)
                curves[name][str(f)] = float(rf.mean())
            log.info("[curve %s %s] %s", fmt, name, {k: round(v, 3) for k, v in curves[name].items()})
        out["curves"][fmt] = curves
    json.dump(out, open(fn, "w"), indent=1)
    torch.save({f"{a[0]}:{a[1]}": circ[a] for a in circ}, R.out / "circuits.pt")
    return out


def stage_items(R, items, behs, actss, fmt, args):
    """Per-item loadings (sum of per-item AP scores over each circuit's units) and per-item
    patch-restored fractions for the numeric circuits (+ random matched controls)."""
    fn = R.out / f"items_{fmt}.json"
    if fn.exists():
        return
    log.info("[items %s]", fmt)
    circ = torch.load(R.out / "circuits.pt")
    names = list(circ)
    union = torch.unique(torch.cat([circ[n] for n in names]))
    pos = {n: torch.searchsorted(union, circ[n]) for n in names}
    beh = behs[fmt]
    ids = _valid_ids(beh)
    loadings = {n: {} for n in names}
    for i in range(0, len(ids), args.bs):
        idx = ids[i:i + args.bs]
        s = ap_batch(R, items, beh, actss[fmt], fmt, idx, units=union.to(R.device))
        for n in names:
            v = s[:, pos[n]].sum(1)
            for j, item_id in enumerate(idx):
                loadings[n][item_id] = float(v[j])
    patch = {}
    for src in ("mattr:numeric", "ap:numeric", f"mattr:{fmt}", f"ap:{fmt}"):
        if src not in circ:
            continue
        rf = restored_fraction(R, items, beh, actss[fmt], fmt, ids, circ[src].to(R.device), args.bs)
        patch[src] = dict(zip(ids, rf.tolist()))
        for r in range(args.n_random):
            u = random_matched(circ[src], R.L, R.d, args.seed + r)
            rf = restored_fraction(R, items, beh, actss[fmt], fmt, ids, u.to(R.device), args.bs)
            patch[f"{src}:random{r}"] = dict(zip(ids, rf.tolist()))
    json.dump({"format": fmt, "ids": ids, "loading": loadings, "patch": patch}, open(fn, "w"))


def stage_itemmattr(R, items, behs, actss, fmt, args):
    """Per-item MAttr: an independent mask per item, trained jointly in batches (per-row k)."""
    fn = R.out / (f"itemmattr_{fmt}" + (f"_{args.item_tag}" if args.item_tag else "") + ".json")
    if fn.exists():
        return
    circ = torch.load(R.out / "circuits.pt")
    names = [n for n in circ if n.split(":")[1] in ("numeric", fmt)]
    beh = behs[fmt]
    recs = {r["id"]: r for r in beh["items"]}
    ids = _valid_ids(beh)
    rng = random.Random(args.seed)
    ids = sorted(rng.sample(ids, min(args.item_n, len(ids))))
    log.info("[itemmattr %s] %d items x %d steps, bs %d", fmt, len(ids), args.item_steps, args.item_bs)
    loadings = {n: {} for n in names}
    top_overlap = {}
    t0 = time.time()
    for i in range(0, len(ids), args.item_bs):
        idx = ids[i:i + args.item_bs]
        B = len(idx)
        scores = torch.zeros(B, R.total, device=R.device, requires_grad=True)
        P = [prompt(items[j], fmt) for j in idx]
        ys = [recs[j]["gen_ids"] for j in idx]; yfs = [recs[j]["gen_flip_ids"] for j in idx]
        cf = actss[fmt]["flip"][idx].to(R.device)
        cf = torch.cat([cf, cf], 0)
        mx = torch.tensor([recs[j]["m_x"] for j in idx], device=R.device)
        mxf = torch.tensor([recs[j]["m_xf"] for j in idx], device=R.device)
        for step in range(args.item_steps):
            logk = torch.rand(B, 1, device=R.device) * math.log(R.total)
            k = torch.exp(logk)
            mask = sigmoid_topk(scores, k, T=args.T, n_iters=30)          # (B, total)
            mm = 1.0 - mask
            R.hook.mode = "blend"
            R.hook.m = torch.cat([mm, mm], 0).view(2 * B, R.L, R.d)
            R.hook.cf = cf
            lp = R.tf_logprob(P + P, ys + yfs)
            R.hook.mode = None
            m = lp[:B] - lp[B:]
            loss = -((m - mxf) / (mx - mxf)).sum()
            g, = torch.autograd.grad(loss, scores)
            with torch.no_grad():
                scores -= args.item_lr * g
        s = scores.detach()
        for n in names:
            v = s[:, circ[n].to(R.device)].sum(1)
            for j, item_id in enumerate(idx):
                loadings[n][item_id] = float(v[j])
        # overlap of each item's own top-1% with the numeric MAttr circuit
        if "mattr:numeric" in circ:
            kk = circ["mattr:numeric"].numel()
            top = torch.topk(s, kk, dim=1).indices.cpu()
            ref = set(circ["mattr:numeric"].tolist())
            for j, item_id in enumerate(idx):
                top_overlap[item_id] = len(ref & set(top[j].tolist())) / kk
        log.info("[itemmattr %s] %d/%d items (%.0fs)  |s|max=%.3g  last loss=%.3f", fmt, i + B,
                 len(ids), time.time() - t0, float(s.abs().max()), float(loss))
    json.dump({"format": fmt, "ids": ids, "loading": loadings, "top_overlap": top_overlap,
               "config": {"steps": args.item_steps, "lr": args.item_lr, "bs": args.item_bs}},
              open(fn, "w"))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--stage", default="all",
                    choices=["behav", "acts", "ap", "mattr", "circuits", "items", "itemmattr", "all"])
    ap.add_argument("--formats", nargs="+", default=list(FORMATS))
    ap.add_argument("--n-items", type=int, default=2000)
    ap.add_argument("--items-seed", type=int, default=0)
    ap.add_argument("--out", default="results/arith_formats")
    ap.add_argument("--bs", type=int, default=16, help="eval batch (items; 2 rows each)")
    ap.add_argument("--gen-bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-margin", type=float, default=1.0,
                    help="min m(x)-m(x') in nats for an item to enter attribution / loading")
    # MAttr (format level)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--train-bs", type=int, default=8)
    ap.add_argument("--variant", default="topk")
    ap.add_argument("--optimizer", default="sgd")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--adam-eps", type=float, default=1e-2)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--loss-norm", action="store_true", help="train on M (rescaled) instead of m (nats)")
    ap.add_argument("--subsets", action="store_true", help="also train numeric MAttr on correct/incorrect items")
    # circuits
    ap.add_argument("--circuit-frac", type=float, default=0.01)
    ap.add_argument("--curve-fracs", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--n-random", type=int, default=1)
    ap.add_argument("--force-circuits", action="store_true")
    # per-item MAttr
    ap.add_argument("--item-n", type=int, default=400)
    ap.add_argument("--item-steps", type=int, default=200)
    ap.add_argument("--item-bs", type=int, default=16)
    ap.add_argument("--item-lr", type=float, default=100.0)
    ap.add_argument("--item-tag", default="", help="suffix for ablation runs of the per-item stage")
    args = ap.parse_args()
    global MIN_MARGIN
    MIN_MARGIN = args.min_margin

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    items_fn = Path(args.out) / f"items_{args.n_items}_{args.items_seed}.json"
    if not items_fn.exists():
        json.dump(make_items(args.n_items, args.items_seed), open(items_fn, "w"))
    items = json.load(open(items_fn))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = Runner(args.model, device, out)
    st = args.stage
    fmts = args.formats

    behs = {f: stage_behav(R, items, f, args.gen_bs) for f in fmts}
    if st == "behav":
        return
    actss = LazyActs(R, items, fmts, args.bs)   # computed on demand, one format in RAM
    if st == "acts":
        for f in fmts:
            actss[f]
        return
    aps = {f: stage_ap(R, items, behs[f], actss[f], f, args.bs) for f in fmts}
    if st == "ap":
        return
    mattrs = {}
    if st in ("mattr", "all", "circuits", "items", "itemmattr"):
        for f in fmts:
            mattrs[f] = stage_mattr(R, items, behs[f], actss[f], f, args)
        if args.subsets and "numeric" in fmts:
            for sub in ("correct", "incorrect"):
                mattrs[f"numeric_{sub}"] = stage_mattr(R, items, behs["numeric"], actss["numeric"],
                                                       "numeric", args, subset=sub)
    if st == "mattr":
        return
    stage_circuits(R, items, behs, actss, aps, mattrs, args)
    if st == "circuits":
        return
    if st in ("items", "all"):
        for f in fmts:
            stage_items(R, items, behs, actss, f, args)
    if st == "itemmattr":            # not part of "all": ~hours per model, run on a few
        for f in fmts:
            stage_itemmattr(R, items, behs, actss, f, args)


if __name__ == "__main__":
    main()

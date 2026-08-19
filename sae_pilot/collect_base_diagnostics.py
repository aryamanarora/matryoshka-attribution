"""FORWARD-ONLY base-point diagnostics for the exact I x G gradient streams.

No SAE. No hook. No backward. No attribution. No MAttr. Nothing is overwritten.

Why a plain base forward is the right object: with error_mode='node' and m = 1,
    new = a_cf + (f_b - f_cf) @ W_dec + [(a_b - dec f_b) - (a_cf - dec f_cf)] = a_b
exactly. So the I x G base point IS the unmodified base forward, and the SAE is not needed
to recover the output distribution there.

Per example we store only scalars. z = final-position logits (float32, as in run_intervened),
p = softmax(z), b = base label, s = source label, d = z_b - z_s.
    grad_z L_LD = -e_b + e_s          (constant, norm sqrt(2))
    grad_z L_CE = p - e_b             (norm shrinks as p_b -> 1; leaks onto all of V)
"""
import os, sys, json, math, time, argparse
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from learning_to_attribute.data.causalgym import CausalGymDataset
from attribute_sae import build_eval_set, sample_train_pair, _key

TASKS = ["npi_ever_subj-relc","npi_any_obj-relc","agr_sv_num_subj-relc",
         "garden_npz_v-trans","garden_npz_obj_mod","filler_gap_pp"]
FIELDS = ["p_base","p_source","p_other","margin","abs_margin","ce_loss","softacc_loss",
          "ce_grad_norm","grad_norm_ratio","ce_ld_grad_cosine","entropy","seq_len"]
SQRT2 = math.sqrt(2.0)


def stream_for(task, seed, n_grad, n_eval=200):
    """Regenerate the exact stream (isolated random.Random(seed); deterministic)."""
    ds = CausalGymDataset("syntaxgym/" + task, seed=seed)
    eval_pairs, eval_keys, eval_draws = build_eval_set(ds, n_eval)
    train, rejected, tkeys = [], 0, set()
    for _ in range(n_grad):
        p, rej = sample_train_pair(ds, eval_keys)
        rejected += rej; tkeys.add(_key(p)); train.append(p)
    inv = {"eval_collection_draws": eval_draws, "n_eval_distinct_keys": len(eval_keys),
           "n_rejected_eval_draws": rejected, "n_train_distinct": len(tkeys),
           "grad_examples_used": len(train), "grad_examples_skipped": 0,
           "train_eval_overlap": len(tkeys & eval_keys)}
    return train, eval_pairs, inv


@torch.no_grad()
def diagnose(model, tokenizer, pairs, device, tag, log_every=1000):
    out = {f: np.zeros(len(pairs), dtype=np.float64) for f in FIELDS}
    t0 = time.time()
    for i, pair in enumerate(pairs):
        base_text = "".join(pair.base_spans)
        ids = tokenizer(base_text, return_tensors="pt").input_ids.to(device)
        b = tokenizer.encode(pair.base_label, add_special_tokens=False)[0]
        s = tokenizer.encode(pair.src_label, add_special_tokens=False)[0]
        z = model(ids).logits[0, -1].float()                 # identical to run_intervened
        p = torch.softmax(z, dim=-1).double()
        pb, ps = p[b].item(), p[s].item()
        d = (z[b] - z[s]).item()
        # g_ce = p - e_b built explicitly; cosine taken from the FULL softmax vector
        g_ce = p.clone(); g_ce[b] -= 1.0
        n_ce = torch.linalg.vector_norm(g_ce).item()
        dot = (-g_ce[b] + g_ce[s]).item()                    # <g_ce, -e_b + e_s>
        out["p_base"][i] = pb; out["p_source"][i] = ps; out["p_other"][i] = 1.0 - pb - ps
        out["margin"][i] = d; out["abs_margin"][i] = abs(d)
        out["ce_loss"][i] = -math.log(max(pb, 1e-300))
        out["softacc_loss"][i] = 1.0 - 1.0 / (1.0 + math.exp(-d)) if d > -700 else 1.0
        out["ce_grad_norm"][i] = n_ce
        out["grad_norm_ratio"][i] = n_ce / SQRT2
        out["ce_ld_grad_cosine"][i] = dot / (n_ce * SQRT2) if n_ce > 0 else float("nan")
        out["entropy"][i] = float(-(p * torch.log(p.clamp_min(1e-300))).sum())
        out["seq_len"][i] = ids.shape[1]
        if log_every and (i + 1) % log_every == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"    [{tag}] {i+1}/{len(pairs)}  {r:.1f} ex/s", flush=True)
    return out


def theory_check(model, tokenizer, pairs, device, n=8):
    """PHASE 3 on REAL logits: autograd vs the closed forms."""
    print("\n=== PHASE 3: closed-form output gradients verified on real model logits ===", flush=True)
    import torch.nn.functional as F
    mx_ce = mx_ld = 0.0; rows = []
    for pair in pairs[:n]:
        ids = tokenizer("".join(pair.base_spans), return_tensors="pt").input_ids.to(device)
        b = tokenizer.encode(pair.base_label, add_special_tokens=False)[0]
        s = tokenizer.encode(pair.src_label, add_special_tokens=False)[0]
        with torch.no_grad():
            z0 = model(ids).logits[0, -1].float()
        z = z0.clone().requires_grad_(True)
        bt, st = torch.tensor([b], device=device), torch.tensor([s], device=device)
        g_ce = torch.autograd.grad(F.cross_entropy(z.unsqueeze(0), bt), z)[0]
        z2 = z0.clone().requires_grad_(True)
        g_ld = torch.autograd.grad(-(z2[bt[0]] - z2[st[0]]), z2)[0]
        p = torch.softmax(z0, -1)
        eb = torch.zeros_like(z0); eb[b] = 1
        es = torch.zeros_like(z0); es[s] = 1
        mx_ce = max(mx_ce, float((g_ce - (p - eb)).abs().max()))
        mx_ld = max(mx_ld, float((g_ld - (-eb + es)).abs().max()))
        rows.append((float(p[b]), float(g_ce.norm()), float(g_ld.norm())))
    print(f"  max|autograd(CE) - (softmax(z) - e_b)| = {mx_ce:.3e}")
    print(f"  max|autograd(LD) - (-e_b + e_s)|       = {mx_ld:.3e}")
    print(f"  {'p_base':>10} {'||g_ce||':>10} {'||g_ld||':>10}")
    for pb, nce, nld in rows:
        print(f"  {pb:>10.6f} {nce:>10.6f} {nld:>10.6f}")
    print("  -> ||g_ld|| is pinned at sqrt(2)=1.414214; ||g_ce|| tracks (1 - p_base).", flush=True)
    return {"max_dev_ce": mx_ce, "max_dev_ld": mx_ld}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--grad-examples", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--results-root", default="results/sae_variant_pilot")
    ap.add_argument("--out", default="results/sae_base_diag")
    args = ap.parse_args()

    assert args.device != "cuda" or torch.cuda.is_available(), "CUDA requested but unavailable"
    print(f"torch {torch.__version__}  device={args.device}  "
          f"cuda={torch.cuda.is_available()}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}",
          flush=True)
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    model.requires_grad_(False)
    print(f"model={args.model} dtype=bfloat16 (matches the I x G runs); NO SAE loaded", flush=True)

    checked = False
    for task in TASKS:
        for seed in (0, 1):
            tag = f"{task}_s{seed}"
            fp = f"{args.out}/{tag}.npz"
            if os.path.exists(fp):
                print(f"[skip] {tag} already collected", flush=True); continue
            train, ev, inv = stream_for(task, seed, args.grad_examples, args.n_eval)
            # self-validate against the ORIGINAL run's recorded invariants
            rp = f"{args.results_root}/{task}/n200/IxG_ce_s{seed}/results.json"
            if os.path.exists(rp):
                prov = json.load(open(rp))["provenance"]
                bad = {k: (v, prov.get(k)) for k, v in inv.items() if prov.get(k) != v}
                assert not bad, f"{tag} stream MISMATCH vs original run: {bad}"
                print(f"[{tag}] stream invariants EXACT vs original run: {inv}", flush=True)
            else:
                print(f"[{tag}] WARNING results.json absent, invariants unverified: {inv}", flush=True)
            if not checked:
                theory_check(model, tokenizer, train, args.device); checked = True
            tr = diagnose(model, tokenizer, train, args.device, tag + ":train")
            evd = diagnose(model, tokenizer, ev, args.device, tag + ":eval", log_every=0)
            np.savez_compressed(fp, **{f"train_{k}": v for k, v in tr.items()},
                                **{f"eval_{k}": v for k, v in evd.items()},
                                meta=np.array([json.dumps({"task": task, "seed": seed, "invariants": inv})]))
            print(f"[done] {tag}  train={len(train)} eval={len(ev)}  -> {fp}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()

"""Decompose behavior into SAE-features vs reconstruction-error, at the intervention layer.
2x2: {features clean|cf} x {error clean|cf}, measured by base-vs-src accuracy & prob-diff.
  (feat=cf, err=cf)   = a_cf                      -> pure counterfactual (lower bound)
  (feat=cf, err=clean)= decode(f_cf)+err_base     -> behavior carried by the ERROR alone
  (feat=clean,err=cf) = decode(f_base)+err_cf     -> behavior carried by FEATURES alone (feature ceiling)
  (feat=clean,err=clean)=a_base                   -> full clean (upper bound)
"""
import argparse, os, sys, json
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from learning_to_attribute.data.causalgym import CausalGymDataset
from scripts.attribute_sae import JumpReLUSAE, strip_bos, pos_map  # reuse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--task", default="syntaxgym/npi_ever_subj-relc")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--sae-repo", default="google/gemma-scope-2b-pt-res")
    ap.add_argument("--sae-id", default="layer_12/width_16k/average_l0_82")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    sae = JumpReLUSAE(args.sae_repo, args.sae_id, dev)
    ds = CausalGymDataset(args.task, seed=7)

    state = {"cf_act": None, "bp": None, "cp": None, "feat": "cf", "err": "cf"}
    def hook(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        if state["cf_act"] is None:  # cache pass
            state["cf_act"] = hs.detach(); return o
        bp = torch.tensor(state["bp"], device=hs.device); cp = torch.tensor(state["cp"], device=hs.device)
        base = hs[0, bp].float(); cf = state["cf_act"][0, cp].float()
        fb, fc = sae.encode(base), sae.encode(cf)
        f = fb if state["feat"] == "clean" else fc
        recon = sae.decode(f)
        err = (base - sae.decode(fb)) if state["err"] == "clean" else (cf - sae.decode(fc))
        hs = hs.clone(); hs[0, bp] = (recon + err).to(hs.dtype)
        return (hs,) + tuple(o[1:]) if isinstance(o, tuple) else hs
    h = model.model.layers[args.layer].register_forward_hook(hook)

    conds = [("feat=cf,err=cf", "cf", "cf"), ("feat=cf,err=clean", "cf", "clean"),
             ("feat=clean,err=cf", "clean", "cf"), ("feat=clean,err=clean", "clean", "clean")]
    res = {c[0]: {"acc": [], "pd": []} for c in conds}
    with torch.no_grad():
        for _ in range(args.n):
            tk = ds.tokenize_pair(strip_bos(ds.sample_pair()), tok, dev)
            bp, cp = pos_map(tk)
            if not bp: continue
            for name, fmode, emode in conds:
                state.update(cf_act=None, bp=bp, cp=cp, feat=fmode, err=emode)
                model(tk.src_input_ids)                       # cache cf
                lg = model(tk.base_input_ids).logits[0, -1].float()
                lp = F.log_softmax(lg, -1)
                res[name]["acc"].append(float(lg[tk.base_label_id] > lg[tk.src_label_id]))
                res[name]["pd"].append((lp[tk.base_label_id] - lp[tk.src_label_id]).item())
    h.remove()
    print(f"=== ERROR vs FEATURE DECOMPOSITION ({args.task}, n={args.n}) ===")
    out = {}
    for name, _, _ in conds:
        a, p = np.mean(res[name]["acc"]), np.mean(res[name]["pd"])
        out[name] = {"acc": float(a), "pd": float(p)}
        print(f"  {name:22s} acc={a:.3f}  probdiff(base-src)={p:+.3f}")
    json.dump(out, open("results/sae_npi_subj_relc/error_decomp.json", "w"), indent=2)

if __name__ == "__main__":
    main()

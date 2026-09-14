"""Per-sign pruning split (stage K) as a markdown table. Prints whatever JSONs exist."""
import json, sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else "results/vw/base")
LAB = {"fisher": "Fisher", "weight_abs": "|W|", "helpfulness": "oracle",
       "E_eps1e-8_lr0.01_s12000_b32": "MAttr b32 12k", "J_eps1e-8_lr0.01_s24000_b64": "MAttr best (b64 24k)",
       "H_eps1e-8_lr0.01_s12000_b32": "MAttr best"}
for fam, stem in (("Tokens→Logits (token rows)", "prune_tok"), ("Features→Logits", "prune_fl")):
    for sgn in ("pos", "neg"):
        p = R / f"{stem}_{sgn}.json"
        if not p.exists():
            print(f"{fam} {sgn}: missing"); continue
        d = json.loads(p.read_text()); full = d["full_loss"]
        print(f"\n**{fam}, {'positive' if sgn=='pos' else 'negative'} weights only** "
              f"({d['n_weights']:,} prunable; all removed: {d['empty_loss']-full:+.4f})\n")
        print("| ranking | " + " | ".join(f"d={x:g}" for x in d["densities"]) + " |")
        print("|---" * (len(d["densities"]) + 1) + "|")
        for k, lab in LAB.items():
            if k in d["curves"]:
                print(f"| {lab} | " + " | ".join(f"{v-full:+.4f}" for v in d["curves"][k]) + " |")

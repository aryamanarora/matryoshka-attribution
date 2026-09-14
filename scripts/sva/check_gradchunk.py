"""Gate for the compute-matched gradient-baseline wave: verify that --grad-batch chunking
reproduces the full-batch scores on the smoke pair in results/_smoke_gradchunk.

Chunking is algebraically exact (scores sum over examples), so the only allowed difference is
fp addition order -- which at bf16 puts ~3% relative noise on individual scores and scrambles
ranks INSIDE the near-zero noise floor (measured on the smoke pair: rank-corr 0.999 in the top
half by |score|, 0.968 in the bottom half, per-unit ratio centred on 1.000). Whole-vector rank
correlation therefore measures the noise tail, not correctness; a real defect (the 3x ragged-
chunk weighting bug this gate caught on 2026-09-06) shows up instead as a std ratio far from 1
and a displaced per-unit ratio. PASS per method = Pearson > 0.999, |std ratio - 1| < 0.05, and
top-1000 overlap > 0.97. Exits nonzero on FAIL or missing files, which holds back every
--dependency=afterok job in the wave. Run from repo root: uv run python scripts/check_gradchunk.py
"""
import glob
import sys

import torch

ok = True
for m in ("ixg", "ig"):
    ga = glob.glob(f"results/_smoke_gradchunk/chunked/*_{m}.scores.pt")
    gb = glob.glob(f"results/_smoke_gradchunk/full/*_{m}.scores.pt")
    if not ga or not gb:
        print(f"{m}: MISSING smoke output ({len(ga)} chunked, {len(gb)} full)")
        ok = False
        continue
    a = torch.load(ga[0], map_location="cpu").float().flatten()
    b = torch.load(gb[0], map_location="cpu").float().flatten()
    pear = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    sr = float(a.std() / b.std().clamp_min(1e-12))
    ov = len(set(torch.topk(a, 1000).indices.tolist())
             & set(torch.topk(b, 1000).indices.tolist())) / 1000
    print(f"{m}: pearson={pear:.6f}  std-ratio={sr:.4f}  top-1000 overlap={ov:.3f}")
    ok = ok and pear > 0.999 and abs(sr - 1) < 0.05 and ov > 0.97
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

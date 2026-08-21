"""Shift and dilation probes for a pixel-level mask -- INCONCLUSIVE, read the caveat.

Pixel-level MAttr (`--methods mattr_pixel`) scores all 224*224 input positions instead of
the 196 patch tokens, and scores ~3x higher than any baseline on the pixel-level
sufficiency sweep. This script was written to test whether that is real selection or an
adversarial artifact (the classic failure of high-capacity perturbation masks, Fong &
Vedaldi 2017), by translating the chosen mask a few pixels: content under it barely
changes, so a mask that selects content should degrade gracefully.

It does NOT settle the question, and the shift numbers should not be quoted as if it did:

  * the mask selects EDGE pixels, and edges are exactly where neighbouring pixels differ
    most, so a one-pixel shift lands across the edge and a collapse is expected even for a
    perfectly legitimate explanation;
  * the intended control (AttnLRP degrading gracefully) is vacuous -- its top-1% mask only
    reaches margin ~0.9 against a clean image of 0.55, so there is nothing to collapse.

The evidence that actually bears on the question points the other way: at 1% of pixels the
model's top-5 is a coherent dog-breed neighbourhood (basenji, dingo, Ibizan hound,
ridgeback), not the junk classes an adversarial pattern produces, and at a matched budget a
score-selected mask beats a dilated one by 2x (14.2 vs 7.5 at 4.5%), i.e. the ranking does
real work. Kept as a probe, not as a verdict.

    .venv-vit/bin/python scripts/vit_teaser_shift_test.py --results results/vit_teaser_pixelate
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_teaser_attr import (make_corrupt_image_sampler, make_target_fn,   # noqa: E402
                             load_model)

SHIFTS = [0, 1, 2, 4, 8, 16]
PIXEL_METHODS = ["mattr_pixel", "attnlrp", "smoothgrad"]


def top_mask(arr, frac):
    """Binary mask of the top `frac` entries of `arr`, same shape."""
    flat = torch.as_tensor(np.ascontiguousarray(arr).ravel().copy())
    k = max(1, round(frac * flat.numel()))
    mask = torch.zeros_like(flat)
    mask[flat.argsort(descending=True)[:k]] = 1.0
    return mask.view(*np.asarray(arr).shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/vit_teaser_pixelate")
    ap.add_argument("--frac", type=float, default=0.01, help="fraction of pixels kept clean")
    ap.add_argument("--draws", type=int, default=32, help="corruption draws averaged")
    args = ap.parse_args()

    res = Path(args.results)
    data = np.load(res / "attributions.npz")
    meta = json.loads((res / "meta.json").read_text())
    a = meta["args"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, weights = load_model(device)
    from PIL import Image
    tf = weights.transforms()
    image = Image.open(a["image"]).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)
    sample_image = make_corrupt_image_sampler(
        x, image, tf, a["baseline"], resample=a["baseline"] in ("shuffle", "pixelate"),
        blur_sigma=a["blur_sigma"], pixel_size=a["pixel_size"], seed=a["seed"] + 1)
    target_fn = make_target_fn(meta["target"], meta["pos"]["index"], meta["neg"]["index"])
    h, w = x.shape[-2:]

    def margin(mask2d):
        m = mask2d.to(device).view(1, 1, h, w)
        with torch.no_grad():
            return float(np.mean([float(target_fn(model(m * x + (1 - m) * sample_image())))
                                  for _ in range(args.draws)]))

    with torch.no_grad():
        clean = float(target_fn(model(x)))
    methods = [m for m in PIXEL_METHODS if f"{m}/pixel" in data.files]
    print(f"explained margin at {args.frac:.1%} of pixels kept clean "
          f"(clean image = {clean:.2f}), mask translated by `shift` pixels")
    print("  shift " + "".join(f"{m:>14}" for m in methods))
    curves = {m: [] for m in methods}
    for s in SHIFTS:
        row = []
        for m in methods:
            v = margin(torch.roll(top_mask(data[f"{m}/pixel"], args.frac), (s, s), (0, 1)))
            curves[m].append(v)
            row.append(v)
        print(f"  {s:>5} " + "".join(f"{v:>14.2f}" for v in row))

    # controls at the same pixel budget
    n_keep = int(top_mask(data[f"{methods[0]}/pixel"], args.frac).sum())
    gen = torch.Generator().manual_seed(0)
    inside = (top_mask(data["mattr/patch"], 0.05)
              .repeat_interleave(16, 0).repeat_interleave(16, 1).flatten().nonzero().squeeze(1))
    ctrl = {}
    for name, pool in [("inside MAttr-patch's top 5% patches", inside),
                       ("uniformly at random", torch.arange(h * w))]:
        m = torch.zeros(h * w)
        m[pool[torch.randperm(len(pool), generator=gen)[:n_keep]]] = 1.0
        ctrl[name] = margin(m.view(h, w))
        print(f"  control: {n_keep} pixels drawn {name}: {ctrl[name]:.2f}")

    out = res / "shift_test.json"
    out.write_text(json.dumps({"frac": args.frac, "shifts": SHIFTS, "clean": clean,
                               "curves": curves, "controls": ctrl}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

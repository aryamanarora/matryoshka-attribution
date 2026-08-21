"""Sufficiency (denoising) curve for every method in the ViT teaser -- the evidence behind
the "faithfulness" row of `plots/plot_vit_teaser.py`.

For each method's patch ranking, keep the top-k% patches CLEAN and corrupt the complement
(MIB's `sufficient` / iso intervention, see CLAUDE.md), then read the explained scalar
logit[dog] - logit[cat]. Sweeping k over the MIB sparsity grid gives a curve; the AUC
reported is its log-spaced area, in raw logit units (see the note next to the printout for
why it is not rescaled so that the clean image reads 1).

Caveat, stated plainly: MAttr *optimises* this objective on this image, so it is expected
to win, and the margin here is not evidence about held-out data -- that is what the MIB
results in the paper are for. What this check rules out is the opposite failure, a headline
panel whose ranking is not even sufficient on its own image.

    .venv-vit/bin/python scripts/vit_teaser_faith.py --results results/vit_teaser_logitdiff
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # run from the repo root
from vit_teaser_attr import (FRACS, N_PATCH, forward_from_patches,   # noqa: E402
                             load_model, make_corrupt_image_sampler, make_target_fn)

METHODS = ["mattr", "mattr_pixel", "attnlrp", "smoothgrad", "gradattnroll",
           "kernelshap"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/vit_teaser_pixelate")
    ap.add_argument("--granularity", default="patch", choices=["patch", "pixel"],
                    help="units the sufficiency intervention acts on. 'patch' compares every "
                         "method on the 196 patch tokens (pixel-native maps are sum-pooled "
                         "into that grid); 'pixel' compares only the methods that are "
                         "natively pixel-level, on all 224*224 positions.")
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--corrupt-samples", type=int, default=16,
                    help="corruption draws averaged per sparsity point (resampled baselines)")
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
    E_clean = model._process_input(x).detach()
    target_fn = make_target_fn(meta["target"], meta["pos"]["index"], meta["neg"]["index"])
    # same corruption distribution the scores were produced against; averaging the sweep over
    # draws keeps one unlucky draw from deciding the ranking
    resample = (meta["baseline"] in ("shuffle", "pixelate", "solid")
                and not a.get("fixed_corrupt", False))
    sample_image = make_corrupt_image_sampler(x, image, tf, meta["baseline"],
                                              resample=resample, blur_sigma=a["blur_sigma"],
                                              pixel_size=a["pixel_size"], seed=a["seed"] + 1)
    n_draw = args.corrupt_samples if resample else 1
    pixel = args.granularity == "pixel"
    h, w = x.shape[-2:]
    total = h * w if pixel else N_PATCH
    key = "pixel" if pixel else "patch"

    def apply_mask(m):
        if pixel:
            mm = m.view(-1, 1, h, w)
            return model(mm * x + (1 - mm) * sample_image())
        mm = m.view(-1, N_PATCH, 1)
        E_corr = model._process_input(sample_image()).detach()
        return forward_from_patches(model, mm * E_clean + (1 - mm) * E_corr)

    def sweep(scores):
        order = torch.as_tensor(np.asarray(scores).ravel(), device=device).argsort(descending=True)
        masks = torch.zeros(len(FRACS), total, device=device)
        for i, f in enumerate(FRACS):
            masks[i, order[:max(1, round(f * total))]] = 1.0
        with torch.no_grad():
            y = torch.stack([target_fn(apply_mask(masks)) for _ in range(n_draw)])
        return y.mean(0).float().cpu().numpy()

    with torch.no_grad():
        y_clean = float(target_fn(forward_from_patches(model, E_clean)))
        y_null = float(np.mean([float(target_fn(model(sample_image()))) for _ in range(n_draw)]))

    rng = np.random.default_rng(0)
    rows = {m: sweep(data[f"{m}/{key}"]) for m in METHODS if f"{m}/{key}" in data.files}
    rows["random"] = np.mean([sweep(rng.standard_normal(total))
                              for _ in range(args.random_seeds)], axis=0)

    # Curves are reported as the RAW explained scalar, not rescaled so that clean = 1. On an
    # image this ambiguous (clean margin 0.55) a good subset beats the clean image by an order
    # of magnitude, and a clean=1 normalisation would turn that into an uninterpretable 20x.
    logf = np.log(FRACS)
    auc = lambda y: np.trapezoid(y, logf) / (logf[-1] - logf[0])   # noqa: E731
    print(f"clean = {y_clean:.3f}, fully corrupted = {y_null:.3f}  "
          f"(explained scalar: {meta['target']}, baseline: {a['baseline']}, "
          f"units: {args.granularity}, n={total})")
    print(f"fraction of {args.granularity}es kept clean ->  " +
          "  ".join(f"{f:>6.1%}" for f in FRACS) + "     AUC")
    for name, y in sorted(rows.items(), key=lambda kv: -auc(kv[1])):
        print(f"  {name:<13} " + "  ".join(f"{v:>6.2f}" for v in y) + f"   {auc(y):>6.3f}")

    out = res / ("faithfulness.json" if not pixel else "faithfulness_pixel.json")
    out.write_text(json.dumps({
        "fracs": FRACS, "clean": y_clean, "corrupted": y_null, "granularity": args.granularity,
        "curves": {k: v.tolist() for k, v in rows.items()},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

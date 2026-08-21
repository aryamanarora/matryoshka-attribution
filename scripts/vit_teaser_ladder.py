"""What the ViT actually sees as the circuit grows, and what it then predicts.

Walks up the sparsity ladder for one method's patch ranking: at each k, the top-k patches
keep their clean content and every other patch is replaced by the corruption (`--baseline`
of the run being read). Saves, for each k, the resulting *image* and the class posterior,
so `plots/plot_vit_sparsity_ladder.py` can show the two side by side. The per-rung masks and
the clean image are saved too, so that figure can alternatively show the kept content alone
rather than composited over the corruption.

The masked image is built in pixel space, but `conv_proj` is applied per 16x16 patch, so
pixel-space masking and the embedding-space masking used for attribution are the same
operation -- the script asserts the two give identical logits rather than assuming it.

With a corruption that is a distribution rather than a fixed image (`--baseline shuffle`),
probabilities are averaged over `--draws` draws, while the displayed image always uses the
canonical member of the family, so the columns differ only in which patches are revealed.

    .venv-vit/bin/python scripts/vit_teaser_ladder.py --method mattr_pixel --granularity pixel
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vit_teaser_attr import (GRID, N_PATCH, PIXELATE_SIZES, corrupt_image,   # noqa: E402
                             forward_from_patches, load_model, patch_shuffle, pixelate)

KS = [1, 2, 4, 8, 16, 32, 64, 128, N_PATCH]           # patch units
PIXEL_FRACS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
PATCH = 16


def masked_image(x, x_corr, mask):
    """[1,3,224,224] image with the mask==0 units taken from the corrupted counterpart.

    `mask` is either 196 patch units (upsampled to the 16x16 blocks) or 224*224 pixel units.
    """
    if mask.numel() == N_PATCH:
        m = (mask.reshape(1, 1, GRID, GRID)
                 .repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3))
    else:
        m = mask.reshape(1, 1, *x.shape[-2:])
    return m * x + (1 - m) * x_corr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/vit_teaser_pixelate")
    ap.add_argument("--method", default="mattr")
    ap.add_argument("--granularity", default="patch", choices=["patch", "pixel"],
                    help="units the ladder reveals; 'pixel' needs a method with a /pixel map")
    ap.add_argument("--topk-print", type=int, default=5,
                    help="how many predicted classes to print per rung")
    ap.add_argument("--draws", type=int, default=32, help="corruption draws averaged")
    ap.add_argument("--out", default=None, help="default: <results>/ladder_<method>.npz")
    args = ap.parse_args()

    res = Path(args.results)
    data = np.load(res / "attributions.npz")
    meta = json.loads((res / "meta.json").read_text())
    a = meta["args"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, weights = load_model(device)
    categories = weights.meta["categories"]
    tf = weights.transforms()
    from PIL import Image
    image = Image.open(a["image"]).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)
    x_corr = corrupt_image(image, tf, x, a["baseline"], blur_sigma=a["blur_sigma"],
                           pixel_size=a["pixel_size"], seed=a["seed"])
    resample = a["baseline"] in ("shuffle", "pixelate", "solid") and not a.get("fixed_corrupt", False)

    pos, neg = meta["pos"]["index"], meta["neg"]["index"]
    pixel = args.granularity == "pixel"
    total = x[0, 0].numel() if pixel else N_PATCH
    key = "pixel" if pixel else "patch"
    scores = torch.as_tensor(data[f"{args.method}/{key}"].ravel().copy(), device=device)
    order = scores.argsort(descending=True)
    ks = ([max(1, round(f * total)) for f in PIXEL_FRACS] if pixel else KS)

    # pixel-space and embedding-space masking must agree, else the pictures below are not
    # the input the attribution was computed on
    E_clean, E_corr = model._process_input(x).detach(), model._process_input(x_corr).detach()
    probe = torch.zeros(N_PATCH, device=device)
    probe[:17] = 1.0        # arbitrary patch subset; `order` may index pixels, not patches
    with torch.no_grad():
        via_pixels = model(masked_image(x, x_corr, probe))
        via_embeds = forward_from_patches(
            model, probe.view(1, -1, 1) * E_clean + (1 - probe.view(1, -1, 1)) * E_corr)
    err = (via_pixels - via_embeds).abs().max().item()
    assert err < 1e-3, f"pixel- and embedding-space masking disagree ({err:.2e})"

    # pixel-space twin of make_corrupt_sampler: the ladder has to *show* the corruption, not
    # just embed it, so it draws from the same families in image space
    gen = torch.Generator().manual_seed(a["seed"] + 7)
    rnd = lambda hi: int(torch.randint(hi, (1,), generator=gen))       # noqa: E731

    def sample_corrupt_image():
        if a["baseline"] == "shuffle":
            return patch_shuffle(x, rnd(10 ** 6))
        size = PIXELATE_SIZES[rnd(len(PIXELATE_SIZES))]
        return pixelate(x, size, (rnd(size), rnd(size)))

    images, masks, probs = [], [], []
    for k in ks:
        mask = torch.zeros(total, device=device)
        mask[order[:k]] = 1.0
        # the mask at PIXEL resolution, so a figure can show the kept content on its own
        # instead of on top of the corruption (see plot_vit_sparsity_ladder --display)
        mask_2d = (mask.reshape(GRID, GRID).repeat_interleave(PATCH, 0)
                       .repeat_interleave(PATCH, 1) if mask.numel() == N_PATCH
                   else mask.reshape(*x.shape[-2:]))
        draws = args.draws if resample else 1
        ps, shown = [], None
        for d in range(draws):
            xc = sample_corrupt_image() if resample else x_corr
            xm = masked_image(x, xc, mask)
            if d == 0:
                # display the *canonical* corruption, not a random draw: columns should
                # differ only in which patches are revealed, not in mosaic coarseness
                shown = masked_image(x, x_corr, mask)
            with torch.no_grad():
                ps.append(model(xm)[0].softmax(-1).cpu())
        images.append(shown[0].cpu().numpy())
        masks.append(mask_2d.cpu().numpy())
        probs.append(torch.stack(ps).mean(0).numpy())

    probs = np.stack(probs)
    mean = np.array(getattr(tf, "mean", [0.485, 0.456, 0.406]))
    std = np.array(getattr(tf, "std", [0.229, 0.224, 0.225]))
    denorm = lambda im: (im.transpose(1, 2, 0) * std + mean).clip(0, 1)   # noqa: E731
    rgb = np.stack([denorm(im) for im in images])
    clean_rgb = denorm(x[0].cpu().numpy())

    top1 = probs.argmax(1)
    out = (Path(args.out) if args.out
           else res / f"ladder_{args.method}{'_pixel' if pixel else ''}.npz")
    np.savez_compressed(
        out, ks=np.array(ks), total=total, granularity=args.granularity,
        images=rgb.astype(np.float32), masks=np.stack(masks).astype(np.uint8),
        clean=clean_rgb.astype(np.float32),
        p_pos=probs[:, pos], p_neg=probs[:, neg], p_top1=probs[np.arange(len(ks)), top1],
        top1=top1, labels=np.array([categories[i] for i in top1]),
        pos_label=meta["pos"]["label"], neg_label=meta["neg"]["label"], method=args.method)

    print(f"{'k':>6} {'frac':>7}  {'p(pos)':>7} {'p(neg)':>7}   top-{args.topk_print}")
    for i, k in enumerate(ks):
        top = np.argsort(-probs[i])[:args.topk_print]
        preds = ", ".join(f"{categories[j]} {probs[i, j]:.2f}" for j in top)
        print(f"{k:>6} {k / total:>7.1%}  {probs[i, pos]:>7.3f} {probs[i, neg]:>7.3f}   {preds}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

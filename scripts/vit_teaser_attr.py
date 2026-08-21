"""Vision-transformer attributions for the AttnLRP-Figure-1-style teaser.

Replicates the qualitative panel of Figure 1 of AttnLRP (Achtibat et al., ICML 2024,
https://openreview.net/forum?id=emtXYlBrNF) -- an image containing a dog and a cat,
explained for "dog" by several attribution methods -- with **MAttr** (soft top-k forward,
log-k schedule, lr=0.05: the headline variant, see CLAUDE.md) as the "ours" panel.

Model: torchvision ViT-B/16, ImageNet-1k weights (the architecture AttnLRP's own ViT
example uses, and the one `lxt` ships LRP rules for).

Attribution substrate. MAttr and KernelSHAP both score the **196 patch tokens** at the
patch-embedding (post `conv_proj`) level: a mask m in [0,1]^196 mixes the clean patch
embeddings with those of a *corrupted* counterpart image (default: each masked 16x16 block
replaced by its own mean colour -- see `--baseline` for why that one),

    E = m * E_clean + (1 - m) * E_corrupt ,

which is exactly MIB/MAttr's `sufficient` (iso / denoising) intervention -- the selected
patches stay CLEAN and the complement is corrupted (see CLAUDE.md). The gradient methods
(AttnLRP, SmoothGrad, Grad x AttnRoll) attribute in their native space; pixel-level maps
are additionally pooled to the same 14x14 patch grid so every panel can be shown either way.

Explained scalar (`--target`), shared by all methods so the panels are comparable:
  logit_diff (default) = logit[pos] - logit[neg]  -- class-contrastive, the thing that makes
                         "explanation for dog" mean dog *rather than* cat;
  single               = logit[pos] alone.
`--explain {dog,cat}` chooses which animal is the positive class, so the same machinery
produces the explanation for either one.

Methods
  mattr         learn_scores(variant="topk", k_schedule="log", lr=0.05, Adam)  [ours]
  mattr_pixel   the same, but over all 224*224 input positions instead of the 196 patches.
                Its sufficiency AUC (10.8) is NOT comparable with the patch-level numbers:
                a 50176-unit mask can keep sub-patch detail, so it is a strictly richer
                intervention space, not a better attribution. Pooled back to patches its
                ranking is mediocre (3.2, vs 5.2 for patch-level MAttr).
  attnlrp       lxt.efficient monkey-patch + zennit Gamma composite, then grad x input.
                NOTE lxt's ViT map is AttnLRP outside attention + CP-LRP inside it -- that
                is upstream's own recommended ViT recipe (lxt/efficient/models/vit_torch.py).
  smoothgrad    mean gradient over `--sg-samples` noisy copies (also stored x input)
  gradattnroll  Chefer et al. (2021) generic attention rollout: R <- R + E_h[(dA * A)^+] R
  kernelshap    KernelSHAP over the 196 patch tokens, antithetic coalition sampling

IMPORTANT (ordering): `lxt.efficient.monkey_patch` rewrites `torch.nn.LayerNorm`/`GELU`/
`MultiheadAttention.forward` **globally and irreversibly** in the process. `attnlrp` is
therefore always run LAST, after every other method, no matter the `--methods` order.

Usage
  uv run --python .venv-vit python scripts/vit_teaser_attr.py \
      --image assets/cat_dog.jpg --out results/vit_teaser
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning_to_attribute.losses import attribution_loss   # noqa: E402
from learning_to_attribute.trainer import learn_scores       # noqa: E402

# ImageNet-1k index ranges. Dogs are 151-268 (Chihuahua .. Mexican hairless);
# domestic cats are 281-285 (tabby, tiger cat, Persian, Siamese, Egyptian).
DOG_IDX = range(151, 269)
CAT_IDX = range(281, 286)
GRID = 14          # 224 / 16
N_PATCH = GRID * GRID
METHOD_ORDER = ["mattr", "mattr_pixel", "smoothgrad", "gradattnroll",
                "kernelshap", "attnlrp"]
# MIB's sparsity grid (fraction of units kept clean), as in results/*/faithfulnesses
FRACS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


# --------------------------------------------------------------------------- model plumbing
def load_model(device):
    from torchvision.models import vision_transformer as vt
    weights = vt.ViT_B_16_Weights.IMAGENET1K_V1
    model = vt.vit_b_16(weights=weights).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, weights


def forward_from_patches(model, patch_emb):
    """Rest of `VisionTransformer.forward` given patch embeddings [B, 196, D]."""
    cls = model.class_token.expand(patch_emb.shape[0], -1, -1)
    z = torch.cat([cls, patch_emb], dim=1)
    z = model.encoder(z)
    return model.heads(z[:, 0])


def patch_shuffle(x, seed=0, patch=16):
    """Randomly permute the 16x16 patches of a [B, 3, H, W] image (deterministic in `seed`)."""
    b, c, h, w = x.shape
    gh, gw = h // patch, w // patch
    perm = torch.randperm(gh * gw, generator=torch.Generator().manual_seed(seed))
    tiles = (x.unfold(2, patch, patch).unfold(3, patch, patch)
              .reshape(b, c, gh * gw, patch, patch))[:, :, perm]
    return (tiles.reshape(b, c, gh, gw, patch, patch)
                 .permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w))


def pixelate(x, size=16, offset=(0, 0)):
    """Replace each `size`x`size` block by its mean colour (mosaic censoring).

    `offset` shifts the block lattice, so varying (size, offset) gives a *family* of mosaics
    of the same image rather than one fixed picture -- see `make_corrupt_sampler`.
    """
    dy, dx = offset
    x = torch.roll(x, shifts=(-dy, -dx), dims=(2, 3))
    y = F.avg_pool2d(x, size).repeat_interleave(size, 2).repeat_interleave(size, 3)
    return torch.roll(y, shifts=(dy, dx), dims=(2, 3))


def solid(x, tf, rgb):
    """A uniform image of colour `rgb` (in [0,1] display space), in the model's input space."""
    mean = torch.tensor(getattr(tf, "mean", [0.485, 0.456, 0.406]), device=x.device)
    std = torch.tensor(getattr(tf, "std", [0.229, 0.224, 0.225]), device=x.device)
    c = (torch.as_tensor(rgb, dtype=x.dtype, device=x.device) - mean) / std
    return c.view(1, 3, 1, 1).expand_as(x).contiguous()


def corrupt_image(image, tf, x, baseline, *, blur_sigma=48.0, pixel_size=16, seed=0):
    """Corrupted counterpart: the "off" state of every masked patch. See `--baseline`."""
    if baseline == "pixelate":
        return pixelate(x, pixel_size)
    if baseline == "solid":     # canonical member of the family: mid grey
        return solid(x, tf, [0.5, 0.5, 0.5])
    if baseline == "shuffle":
        return patch_shuffle(x, seed)
    if baseline == "blur":
        # blur in the model's own 224px space, so sigma means the same thing for any source
        # image; the PIL-side radius it replaced was in original-resolution pixels
        k = int(4 * blur_sigma) | 1
        return TF.gaussian_blur(x, [k, k], [float(blur_sigma)] * 2)
    if baseline in ("black", "white"):
        v = 0 if baseline == "black" else 255
        return tf(Image.new("RGB", image.size, (v, v, v))).unsqueeze(0).to(x.device)
    return torch.zeros_like(x)              # dataset mean == 0 after normalisation


PIXELATE_SIZES = (8, 16, 32)


def make_corrupt_image_sampler(x, image, tf, baseline, *, resample=True, blur_sigma=48.0,
                               pixel_size=16, seed=0):
    """Returns `sample() -> corrupted image [1, 3, 224, 224]`.

    With `resample`, every call draws a *different* corrupted counterpart from a family
    rather than reusing one frozen image. This matters, and measurably: MIB/MAttr trains
    against a distribution of counterfactual sources (a new (base, source) pair every step),
    and against a single frozen source a patch's score partly measures the accident of that
    one source. The families are
      shuffle  -- a fresh permutation of the image's own patches;
      pixelate -- a fresh (block size, lattice offset) mosaic of the image;
      solid    -- a fresh uniform-random colour, filling the whole frame.
    `solid` exists to separate two properties that `pixelate` conflates: being a
    *distribution* rather than one frozen image, and being *structured* (each masked unit
    keeps its own local colour) rather than a flat out-of-distribution hole. It is a
    distribution of flat holes, so comparing it against fixed `white` and against
    `pixelate` (with and without `--fixed-corrupt`) isolates which property matters.
    The remaining baselines are single images and are always used fixed.
    """
    fixed = corrupt_image(image, tf, x, baseline, blur_sigma=blur_sigma,
                          pixel_size=pixel_size, seed=seed)
    if not resample or baseline not in ("shuffle", "pixelate", "solid"):
        return lambda: fixed

    gen = torch.Generator().manual_seed(seed)
    rnd = lambda hi: int(torch.randint(hi, (1,), generator=gen))   # noqa: E731
    if baseline == "shuffle":
        return lambda: patch_shuffle(x, rnd(10 ** 6))
    if baseline == "solid":
        return lambda: solid(x, tf, torch.rand(3, generator=gen).tolist())

    def sample_pixelate():
        size = PIXELATE_SIZES[rnd(len(PIXELATE_SIZES))]
        return pixelate(x, size, (rnd(size), rnd(size)))

    return sample_pixelate


def make_corrupt_sampler(model, x, image, tf, baseline, **kw):
    """`sample() -> corrupted patch embeddings [1, 196, D]`; embeds the image sampler.

    Permuting patch embeddings is exactly permuting image patches (conv_proj is per patch),
    so the embedding view and the pixel view of a corruption are the same object.
    """
    sample_image = make_corrupt_image_sampler(x, image, tf, baseline, **kw)
    return lambda: model._process_input(sample_image()).detach()


def make_auc_probe(forward_masked, total, n_draw=4):
    """`probe(scores) -> (sufficiency AUC, curve)` at the fixed FRACS grid.

    The training loss alone is not a convergence signal: k is resampled every step, so the
    loss trace mostly reflects which k was drawn. This evaluates the *current ranking* under
    the hard top-k intervention at every sparsity in FRACS and returns the log-spaced area --
    the same quantity `vit_teaser_faith.py` reports, and the thing actually being optimised.

    `forward_masked(masks [F, total]) -> [F]` applies a batch of hard masks and returns the
    explained scalar, so the same probe serves patch- and pixel-level units.
    """
    logf = np.log(FRACS)
    keep = [max(1, round(f * total)) for f in FRACS]

    def probe(scores):
        order = scores.detach().argsort(descending=True)
        masks = torch.zeros(len(FRACS), total, device=scores.device)
        for i, k in enumerate(keep):
            masks[i, order[:k]] = 1.0
        with torch.no_grad():
            y = torch.stack([forward_masked(masks) for _ in range(n_draw)])
        curve = y.mean(0).float().cpu().numpy()
        return float(np.trapezoid(curve, logf) / (logf[-1] - logf[0])), curve

    return probe


def make_target_fn(kind, pos, neg):
    """logits [B, 1000] -> [B] scalar being explained.

    `pos` is the class the figure explains, `neg` the class it is explained *against*.
    Which animal plays which role is set by `--explain`, so nothing downstream assumes the
    positive class is the dog.
    """
    if kind == "single":
        return lambda y: y[:, pos]
    return lambda y: y[:, pos] - y[:, neg]


def pool_to_patches(pix):
    """[224, 224] pixel map -> [14, 14] patch map (sum of relevance inside each patch)."""
    h, w = pix.shape
    return pix.reshape(GRID, h // GRID, GRID, w // GRID).sum(axis=(1, 3))


# --------------------------------------------------------------------------- methods
def run_mattr(model, x, E_clean, sample_image, target_fn, pos, neg, args, device, *,
              units="patch"):
    """MAttr headline variant: soft top-k forward, log-k schedule, Adam, lr 0.05.

    `units="patch"` scores the 196 patch tokens, masking at the patch-embedding level.
    `units="pixel"` scores all 224*224 input positions instead, masking the image itself
    before `conv_proj` (one score per position, shared across RGB -- a per-channel mask
    would attribute to colour channels, which is not what the figure is about). Everything
    else -- soft top-k forward, log-uniform k, Adam, lr, T -- is identical, so the two runs
    differ only in the granularity of the attribution variable.
    """
    base_id = torch.tensor([pos], device=device)
    source_id = torch.tensor([neg], device=device)
    loss_name = "logit" if args.target == "single" else "logit_diff"
    pixel = units == "pixel"
    h, w = x.shape[-2:]
    total = h * w if pixel else N_PATCH

    def apply_mask(m):
        """m: [B, total] -> logits [B, 1000]."""
        if pixel:
            mm = m.view(-1, 1, h, w)
            return model(mm * x + (1 - mm) * sample_image())
        mm = m.view(-1, N_PATCH, 1)
        E_corr = model._process_input(sample_image()).detach()
        return forward_from_patches(model, mm * E_clean + (1 - mm) * E_corr)

    def loss_fn(mask):
        # corrupt_topk=False == sufficient / iso: the top-k units stay clean and must retain
        # the base behaviour (see learning_to_attribute.losses).
        return attribution_loss(loss_name, apply_mask(mask.unsqueeze(0)),
                                base_id, source_id, corrupt_topk=False)

    probe = make_auc_probe(lambda masks: target_fn(apply_mask(masks)), total, args.probe_draws)
    trace = []

    def on_step(step, k, loss, scores):
        if args.probe_every and (step % args.probe_every == 0 or step == args.steps - 1):
            auc, _ = probe(scores)
            trace.append((step, auc))
            print(f"  probe step {step:5d}  sufficiency AUC = {auc:7.3f}", flush=True)

    res = learn_scores(
        total, loss_fn,
        steps=args.steps, variant="topk", k_schedule="log", lr=args.lr,
        optimizer="adam", T=args.T, n_iters=args.n_iters, device=device,
        log_every=args.log_every, logger=_LOG, on_step=on_step,
    )
    scores = res.scores.numpy()
    out = {"loss_log": np.asarray(res.loss_log, dtype=np.float32),
           "k_log": np.asarray(res.k_log, dtype=np.float32),
           "auc_trace": np.asarray(trace, dtype=np.float32).reshape(-1, 2)}
    out["pixel" if pixel else "patch"] = scores.reshape(h, w) if pixel else \
        scores.reshape(GRID, GRID)
    return out


def run_smoothgrad(model, x, target_fn, args, device):
    g = torch.zeros_like(x)
    sigma = args.sg_noise * float(x.max() - x.min())
    gen = torch.Generator(device=device).manual_seed(args.seed)
    for i in range(0, args.sg_samples, args.batch_size):
        b = min(args.batch_size, args.sg_samples - i)
        noise = torch.randn(b, *x.shape[1:], device=device, generator=gen) * sigma
        xb = (x + noise).requires_grad_(True)
        target_fn(model(xb)).sum().backward()
        g += xb.grad.sum(0, keepdim=True)
    g /= args.sg_samples
    return {"pixel": g.sum(1)[0].detach().cpu().numpy(),
            "pixel_xinput": (g * x).sum(1)[0].detach().cpu().numpy()}


def run_gradattnroll(model, x, target_fn, device, atol=1e-3):
    """Chefer et al. (2021) generic attention explainability (grad-weighted rollout).

    `nn.MultiheadAttention(..., need_weights=True)` returns a *view* of the attention
    probabilities that is off the path to the output, so `retain_grad` on it yields None.
    We therefore re-implement the (unmasked, eval-mode) MHA inline from the module's own
    weights and keep a handle on the exact softmax tensor that feeds the value product.
    The reimplementation is checked against the stock forward before it is used.
    """
    import math

    from torchvision.models.vision_transformer import EncoderBlock
    attns = []
    original_forward = EncoderBlock.forward

    def capturing_forward(self, input):
        mha = self.self_attention
        h = self.ln_1(input)
        b, s, d = h.shape
        nh = mha.num_heads
        hd = d // nh
        q, k, v = F.linear(h, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.view(b, s, nh, hd).transpose(1, 2) for t in (q, k, v))
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(hd), dim=-1)
        attn.retain_grad()
        attns.append(attn)
        o = mha.out_proj((attn @ v).transpose(1, 2).reshape(b, s, d))
        h = self.dropout(o) + input
        return self.mlp(self.ln_2(h)) + h

    with torch.no_grad():
        reference = model(x)
    EncoderBlock.forward = capturing_forward
    try:
        xi = x.clone().requires_grad_(True)   # params are frozen: anchor the graph on input
        logits = model(xi)
        err = (logits - reference).abs().max().item()
        assert err < atol, f"inlined attention diverges from the stock forward ({err:.2e})"
        target_fn(logits).sum().backward()
    finally:
        EncoderBlock.forward = original_forward

    s = attns[0].shape[-1]
    R = torch.eye(s, device=device)
    for a in attns:
        abar = (a.grad * a).clamp(min=0).mean(1)[0]     # mean over heads -> [S, S]
        R = R + abar @ R
    return {"patch": R[0, 1:].reshape(GRID, GRID).detach().cpu().numpy()}


def run_kernelshap(model, E_clean, sample_corrupt, target_fn, args, device):
    """KernelSHAP over the 196 patch tokens, on the same masking substrate as MAttr.

    Coalition sizes are drawn proportional to the Shapley kernel pi(s) = (n-1)/(s(n-s)), so
    the sampled design is already kernel-weighted and an *unweighted* ridge fit recovers the
    SHAP values. Samples are antithetic (z and its complement) for variance reduction. The
    "absent" state is drawn from the same corruption sampler MAttr trains against, which is
    also SHAP's own convention of marginalising over a background distribution.
    """
    n = N_PATCH
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    s = torch.arange(1, n, dtype=torch.float64)
    p = ((n - 1) / (s * (n - s)))
    p = (p / p.sum()).float()

    m_half = args.shap_samples // 2
    sizes = torch.multinomial(p, m_half, replacement=True, generator=gen) + 1
    Z = torch.zeros(2 * m_half, n)
    for i, k in enumerate(sizes.tolist()):
        idx = torch.randperm(n, generator=gen)[:k]
        Z[2 * i, idx] = 1.0
        Z[2 * i + 1] = 1.0 - Z[2 * i]           # antithetic partner

    ys, nulls = [], []
    with torch.no_grad():
        for i in range(0, Z.shape[0], args.batch_size):
            m = Z[i:i + args.batch_size].to(device).unsqueeze(-1)
            E = m * E_clean + (1 - m) * sample_corrupt()
            ys.append(target_fn(forward_from_patches(model, E)).float().cpu())
        y_full = target_fn(forward_from_patches(model, E_clean)).float().cpu()
        for _ in range(args.null_samples):
            nulls.append(target_fn(forward_from_patches(model, sample_corrupt())).float().cpu())
    y = torch.cat(ys)
    y_null = torch.stack(nulls).mean(0)         # E_background[f(empty coalition)]

    # Enforce efficiency (sum phi = f(1) - f(0)) by eliminating the last coefficient:
    #   y - f(0) - z_n * (f(1) - f(0)) = sum_{j<n} (z_j - z_n) phi_j
    Zc = Z[:, :-1] - Z[:, -1:]
    tgt = (y - y_null) - Z[:, -1] * (y_full - y_null)
    A = Zc.double().T @ Zc.double() + args.shap_ridge * torch.eye(n - 1, dtype=torch.float64)
    phi = torch.linalg.solve(A, Zc.double().T @ tgt.double())
    phi = torch.cat([phi, (y_full - y_null).double() - phi.sum()])
    return {"patch": phi.float().numpy().reshape(GRID, GRID)}


def run_attnlrp(model_builder, x, target_fn, args, device):
    """AttnLRP via lxt + a zennit Gamma composite. Patches torch.nn globally -- run last."""
    import zennit.rules as z_rules
    from torchvision.models import vision_transformer as vt
    from zennit.composites import LayerMapComposite

    from lxt.efficient import monkey_patch, monkey_patch_zennit

    monkey_patch(vt, verbose=False)
    monkey_patch_zennit(verbose=False)
    model, _ = model_builder(device)         # rebuild so the patched classes are in effect

    comp = LayerMapComposite([
        (torch.nn.Conv2d, z_rules.Gamma(args.lrp_conv_gamma)),
        (torch.nn.Linear, z_rules.Gamma(args.lrp_lin_gamma)),
    ])
    comp.register(model)
    xi = x.clone().requires_grad_(True)
    target_fn(model(xi)).sum().backward()
    comp.remove()
    return {"pixel": (xi * xi.grad).sum(1)[0].detach().cpu().numpy()}


# --------------------------------------------------------------------------- driver
class _Log:
    def info(self, fmt, *a):
        print(fmt % a, flush=True)


_LOG = _Log()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default="assets/cat_dog.jpg")
    ap.add_argument("--out", default="results/vit_teaser")
    ap.add_argument("--methods", default="all",
                    help=f"comma-separated subset of {METHOD_ORDER}, or 'all'")
    ap.add_argument("--target", default="logit_diff", choices=["logit_diff", "single"],
                    help="scalar every method explains: the contrast between the two classes "
                         "(default) or the explained class's logit alone")
    ap.add_argument("--explain", default="dog", choices=["dog", "cat"],
                    help="which animal is the POSITIVE class of the explained contrast; "
                         "'cat' explains the cat against the dog instead")
    ap.add_argument("--dog-class", type=int, default=None, help="default: argmax over dogs")
    ap.add_argument("--cat-class", type=int, default=None, help="default: argmax over cats")
    ap.add_argument("--baseline", default="pixelate",
                    choices=["pixelate", "shuffle", "solid", "blur", "mean", "black",
                             "white"],
                    help="corrupted counterpart for the mask-based methods. A corruption has "
                         "to be BOTH class-neutral and something the model can still read as "
                         "an image. Measured on this image (clean: dog 8.14 / cat 7.59): "
                         "'pixelate' (default) replaces each masked 16x16 block by its own "
                         "mean colour -- dog -0.34 / cat -0.31, i.e. neutral to within 0.03 "
                         "logits, while each masked region keeps its local colour instead of "
                         "becoming a uniform hole. 'shuffle' permutes the image's own patches "
                         "(dog 2.77 / cat 1.77) -- also workable, but the jigsaw texture is "
                         "read as its own class. 'blur' at sigma 48 is neutral (diff 0.02) but "
                         "at smaller sigma is NOT: it wipes the Siamese cat's face while "
                         "leaving a tan dog-shaped blob, so the corrupted image predicts dog "
                         "by ~6 logits and sufficiency is maximised by deleting the cat "
                         "rather than by keeping the dog. 'mean'/'black' are neutral but turn "
                         "every masked patch into the same flat hole.")
    ap.add_argument("--blur-sigma", type=float, default=48.0,
                    help="Gaussian sigma in 224px model space for --baseline blur")
    ap.add_argument("--pixel-size", type=int, default=16,
                    help="block size for --baseline pixelate (16 = one ViT patch)")
    # MAttr (headline variant: soft top-k forward + log-k schedule + Adam, lr 0.05)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--n-iters", type=int, default=30)
    # baselines
    ap.add_argument("--sg-samples", type=int, default=256)
    ap.add_argument("--sg-noise", type=float, default=0.15)
    ap.add_argument("--shap-samples", type=int, default=16384)
    ap.add_argument("--shap-ridge", type=float, default=1.0)
    ap.add_argument("--null-samples", type=int, default=32,
                    help="draws used to estimate f(empty coalition) when resampling")
    ap.add_argument("--fixed-corrupt", action="store_true",
                    help="reuse ONE corrupted counterpart for every masked forward instead of "
                         "resampling it (only meaningful for --baseline shuffle)")
    ap.add_argument("--lrp-conv-gamma", type=float, default=0.25)
    ap.add_argument("--lrp-lin-gamma", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--probe-every", type=int, default=25,
                    help="steps between sufficiency-AUC probes during MAttr training (0=off)")
    ap.add_argument("--probe-draws", type=int, default=8,
                    help="corruption draws averaged per probe")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, weights = load_model(device)
    categories = weights.meta["categories"]
    tf = weights.transforms()
    image = Image.open(args.image).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)

    x_corr = corrupt_image(image, tf, x, args.baseline, blur_sigma=args.blur_sigma,
                           pixel_size=args.pixel_size, seed=args.seed)

    with torch.no_grad():
        logits = model(x)[0]
    dog = args.dog_class if args.dog_class is not None else max(DOG_IDX, key=lambda i: logits[i])
    cat = args.cat_class if args.cat_class is not None else max(CAT_IDX, key=lambda i: logits[i])
    pos, neg = (dog, cat) if args.explain == "dog" else (cat, dog)
    print(f"explaining {categories[pos]} ({logits[pos]:.3f}) against "
          f"{categories[neg]} ({logits[neg]:.3f})", flush=True)
    target_fn = make_target_fn(args.target, pos, neg)

    E_clean = model._process_input(x).detach()
    E_corr = model._process_input(x_corr).detach()
    # shuffle and pixelate are families to draw from; the others are a single fixed image
    resample = args.baseline in ("shuffle", "pixelate", "solid") and not args.fixed_corrupt
    sample_image = make_corrupt_image_sampler(x, image, tf, args.baseline,
                                              resample=resample, blur_sigma=args.blur_sigma,
                                              pixel_size=args.pixel_size, seed=args.seed)
    sample_corrupt = lambda: model._process_input(sample_image()).detach()   # noqa: E731
    print(f"corruption: {args.baseline}" + (" (resampled per forward)" if resample else ""),
          flush=True)

    wanted = METHOD_ORDER if args.methods == "all" else args.methods.split(",")
    unknown = set(wanted) - set(METHOD_ORDER)
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}; known: {METHOD_ORDER}")
    # attnlrp last: lxt's monkey_patch mutates torch.nn classes irreversibly (see docstring)
    todo = [m for m in METHOD_ORDER if m in wanted]

    arrays, timings = {}, {}
    for name in todo:
        t0 = time.time()
        print(f"[{name}] ...", flush=True)
        if name in ("mattr", "mattr_pixel"):
            res = run_mattr(model, x, E_clean, sample_image, target_fn, pos, neg, args, device,
                            units="pixel" if name == "mattr_pixel" else "patch")
        elif name == "smoothgrad":
            res = run_smoothgrad(model, x, target_fn, args, device)
        elif name == "gradattnroll":
            res = run_gradattnroll(model, x, target_fn, device)
        elif name == "kernelshap":
            res = run_kernelshap(model, E_clean, sample_corrupt, target_fn, args, device)
        else:
            res = run_attnlrp(load_model, x, target_fn, args, device)
        timings[name] = time.time() - t0
        for k, v in res.items():
            arrays[f"{name}/{k}"] = v
        if "pixel" in res:                                # common 14x14 view for every panel
            arrays[f"{name}/patch"] = pool_to_patches(res["pixel"])
        print(f"[{name}] done in {timings[name]:.1f}s", flush=True)

    mean = np.array(getattr(tf, "mean", [0.485, 0.456, 0.406]))
    std = np.array(getattr(tf, "std", [0.229, 0.224, 0.225]))
    rgb = (x[0].cpu().numpy().transpose(1, 2, 0) * std + mean).clip(0, 1)
    arrays["input/rgb"] = rgb.astype(np.float32)

    np.savez_compressed(out / "attributions.npz", **arrays)
    (out / "meta.json").write_text(json.dumps({
        "image": args.image, "target": args.target, "baseline": args.baseline,
        "explain": args.explain,
        "pos": {"index": pos, "label": categories[pos], "logit": float(logits[pos])},
        "neg": {"index": neg, "label": categories[neg], "logit": float(logits[neg])},
        "seconds": timings, "device": device, "args": vars(args),
    }, indent=2))
    print(f"wrote {out}/attributions.npz  ({', '.join(sorted(arrays))})")


if __name__ == "__main__":
    main()

"""Datasets, models, oracles and attribution methods for the Bilodeau et al. (2022)
impossibility benchmark ("Impossibility Theorems for Feature Attribution", arXiv:2212.11870).

Port of `google-research/interpretability-theory` (archived, TF2) to torch, so MAttr
(`learning_to_attribute.trainer.learn_scores`) can be run as one more arm alongside their
SHAP / IG / LIME / SmoothGrad / Gradient. Ported faithfully: the five UCI preprocessing
recipes (`datasets.py`), the five model architectures and their epoch/batch settings
(`models.py`, `run_tabular_experiment.py`), the +-10%-of-range 20-step perturbation grid,
the recourse and spurious ORACLES, and the per-example min-max `_normalize` to [-1, 1] that
makes the threshold sweep comparable across methods (`tabular_experiment.py`).

Deliberate deviations, all noted where they occur:
  * KernelSHAP is reimplemented (~40 lines) rather than pulled from `shap`, which is not in
    this venv -- same configuration they used (100 background rows, 500 subsets).
  * `sklearn.model_selection.train_test_split` -> a numpy permutation split.
  * their `chess.csv` is read with a header row (so krkopt loses its first game); we read
    with `header=None`. One row out of 28k.

Everything the oracle touches is in RAW feature space and standardized on the way into the
model, exactly as in `tabular_experiment.perturbation`.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------- datasets

DATA_URLS = {
    "wine": "wine/wine.data",
    "ecoli": "ecoli/ecoli.data",
    "abalone": "abalone/abalone.data",
    "credit": "credit-screening/crx.data",
    "chess": "chess/king-rook-vs-king/krkopt.data",
}
UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

# name -> (epochs, batch_size), from the commented-out train_models calls in
# run_tabular_experiment.py.
TRAIN_CFG = {
    "wine": (50, 16), "ecoli": (100, 32), "abalone": (20, 64),
    "credit": (50, 64), "chess": (50, 1000),
}


def load_tabular(name, root="data/impossibility", seed=0):
    """-> (features [N, p] raw, labels [N], ordered_feature_idxs, standardize, task).

    `ordered_feature_idxs` are the non-categorical features the oracle is evaluated on;
    `task` is "multiclass" | "binary" | "regression" and decides the head and the loss.
    """
    rng = np.random.default_rng(seed)
    path = Path(root) / Path(DATA_URLS[name]).name

    if name == "wine":
        d = pd.read_csv(path, header=None)
        labels = d[0].to_numpy() - 1
        feats = d.iloc[:, 1:14].to_numpy(float)
        ordered, task = list(range(13)), "multiclass"
    elif name == "ecoli":
        d = pd.read_csv(path, header=None, sep=r"\s+").iloc[:, 1:]
        d[8] = pd.factorize(d[8])[0]
        labels = d[8].to_numpy()
        feats = d.drop(8, axis=1).to_numpy(float)
        ordered, task = list(range(7)), "multiclass"
    elif name == "abalone":
        d = pd.read_csv(path, header=None)
        d[0] = d[0].replace(["M", "F", "I"], [1, -1, 0])
        labels = d[8].to_numpy(float)
        feats = d.drop(8, axis=1).to_numpy(float)
        ordered, task = [1, 2, 3, 4, 5, 6, 7], "regression"
    elif name == "credit":
        d = pd.read_csv(path, header=None).replace("?", np.nan)
        for c in (1, 13):
            d[c] = pd.to_numeric(d[c])
        for c in (0, 3, 4, 5, 6):
            d[c] = d[c].fillna(d[c].mode()[0])
        for c in (1, 13):
            d[c] = d[c].fillna(d[c].mean())
        for c in (0, 3, 4, 5, 6, 8, 9, 11, 12, 15):
            d[c] = pd.factorize(d[c])[0]
        labels = d[15].to_numpy(float)
        feats = d.drop(15, axis=1).to_numpy(float)
        ordered, task = [1, 2, 7, 10, 13, 14], "binary"
    elif name == "chess":
        d = pd.read_csv(path, header=None)
        for c in (0, 2, 4):
            d[c] = pd.factorize(d[c])[0]
        labels = (d[6] != "draw").to_numpy(float)
        feats = d.iloc[:, :6].to_numpy(float)
        ordered, task = list(range(6)), "binary"
    else:
        raise ValueError(f"unknown dataset {name!r}")

    perm = rng.permutation(len(feats))
    feats, labels = feats[perm], labels[perm]
    mean, std = feats.mean(0), feats.std(0)
    std = np.where(std < 1e-12, 1.0, std)

    def standardize(x):
        return (x - mean) / std

    return feats, labels, ordered, standardize, task


# ---------------------------------------------------------------- models

class TabularNet(nn.Module):
    """Torch port of `models.create_tabular_model`. `forward` returns the same thing the
    keras model did: class PROBABILITIES for the softmax/sigmoid heads, the raw relu value
    for abalone's regression head. Shape [B, C] with C=1 for binary/regression, so
    `argmax(-1)` reproduces their `max_class` in every case."""

    def __init__(self, name, p):
        super().__init__()
        if name == "wine":
            body = [nn.Linear(p, 12), nn.Linear(12, 24), nn.ReLU(),
                    nn.Linear(24, 8), nn.ReLU(), nn.Dropout(0.2), nn.Linear(8, 3)]
            self.head = "softmax"
        elif name == "ecoli":
            body = [nn.Linear(p, 12), nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 8)]
            self.head = "softmax"
        elif name == "abalone":
            body = [nn.Linear(p, 256), nn.ReLU(), nn.BatchNorm1d(256),
                    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3), nn.BatchNorm1d(256),
                    nn.Linear(256, 1), nn.ReLU()]
            self.head = "identity"
        elif name == "credit":
            body = [nn.Linear(p, 256), nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)]
            self.head = "sigmoid"
        elif name == "chess":
            body = [nn.Linear(p, 256), nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)]
            self.head = "sigmoid"
        else:
            raise ValueError(name)
        self.net = nn.Sequential(*body)

    def forward(self, x):
        z = self.net(x)
        if self.head == "softmax":
            return z.softmax(-1)
        if self.head == "sigmoid":
            return z.sigmoid()
        return z


def train_model(name, feats, labels, standardize, task, seed, device="cpu"):
    """80/20 split, Adam, the epochs/batch of run_tabular_experiment.py. -> (model, test_acc)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    epochs, bs = TRAIN_CFG[name]
    n = len(feats)
    perm = rng.permutation(n)
    n_te = int(0.2 * n)
    te, tr = perm[:n_te], perm[n_te:]
    X = torch.tensor(standardize(feats), dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32 if task != "multiclass" else torch.long,
                     device=device)

    model = TabularNet(name, feats.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters())
    for _ in range(epochs):
        model.train()
        idx = tr[rng.permutation(len(tr))]
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            opt.zero_grad()
            z = model.net(X[b])
            if task == "multiclass":
                loss = nn.functional.cross_entropy(z, y[b])
            elif task == "binary":
                loss = nn.functional.binary_cross_entropy_with_logits(z[:, 0], y[b])
            else:
                loss = nn.functional.mse_loss(z.relu()[:, 0], y[b])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        z = model.net(X[te])
        if task == "multiclass":
            score = (z.argmax(-1) == y[te]).float().mean().item()
        elif task == "binary":
            score = ((z[:, 0] > 0).float() == y[te]).float().mean().item()
        else:  # report R^2 so a single "quality" column stays readable
            pred = z.relu()[:, 0]
            score = 1 - ((pred - y[te]) ** 2).sum().item() / ((y[te] - y[te].mean()) ** 2).sum().item()
    return model, score


class CountingModel:
    """Wraps a trained net as a plain `f(np.ndarray [B, p] RAW) -> np.ndarray [B, C]` and
    counts forward-passed ROWS. Row count is the query budget every method is charged in,
    which is the axis the paper's Theorem 5.1 lower bound lives on."""

    def __init__(self, model, standardize, device="cpu"):
        self.model, self.standardize, self.device = model, standardize, device
        self.rows = 0
        model.eval()

    def reset(self):
        self.rows = 0

    def torch_std(self, x_std):
        """Differentiable path: takes an ALREADY-standardized torch tensor."""
        self.rows += x_std.shape[0]
        return self.model(x_std)

    def to_std(self, x_raw):
        return torch.tensor(self.standardize(np.atleast_2d(x_raw)),
                            dtype=torch.float32, device=self.device)

    def __call__(self, x_raw):
        with torch.no_grad():
            return self.torch_std(self.to_std(x_raw)).cpu().numpy()


# ---------------------------------------------------------------- oracles

N_STEPS = 20            # tabular_experiment.perturbation, "this should be even"
PERCENT_PERTURB = 0.1


def feature_ranges(feats):
    return feats.max(0) - feats.min(0)


def perturbation(ranges, example, j, n_steps=N_STEPS):
    """Exact port: n_steps copies of `example` with feature j swept over
    +-`PERCENT_PERTURB` * (data range of j), linearly spaced."""
    delta = np.linspace(-PERCENT_PERTURB * ranges[j], PERCENT_PERTURB * ranges[j], n_steps)
    out = np.tile(example, (n_steps, 1))
    out[:, j] = example[j] + delta
    return out


def recourse_oracle(out):
    """1 if the mean output over the UPPER half of the perturbation grid exceeds the
    lower half -- "does increasing this feature increase the output"."""
    h = len(out) // 2
    return float(out[h:].mean() > out[:h].mean())


def spurious_quantile(f, ranges, examples, ordered, q=0.8):
    """The paper's model-dependent oracle threshold: the 80th quantile of the
    perturbation-output variance over 100 reference examples x all ordered features."""
    v = []
    for ex in examples:
        c = int(np.argmax(f(ex.reshape(1, -1))))
        for j in ordered:
            v.append(np.var(f(perturbation(ranges, ex, j))[:, c]))
    return float(np.quantile(v, q))


# ---------------------------------------------------------------- attribution methods

def normalize(values):
    """Port of `_normalize`: per-example min-max onto [-1, 1]. This is why the paper's
    threshold sweep over [-1, 1] is a sweep over WITHIN-EXAMPLE rank rather than absolute
    score, and why methods on wildly different scales are still comparable."""
    lo, hi = np.min(values), np.max(values)
    return 2 * (values - lo) / max(hi - lo, 1e-10) - 1


def _grad_of_class(cm, x_std, c):
    x = x_std.clone().requires_grad_(True)
    cm.torch_std(x)[:, c].sum().backward()
    return x.grad.detach().cpu().numpy()


def m_gradient(cm, x_raw, c, **_):
    return _grad_of_class(cm, cm.to_std(x_raw), c)[0]


def m_smoothgrad(cm, x_raw, c, n_iters=100, local_radius=0.1, rng=None, **_):
    x = cm.to_std(x_raw)
    noise = torch.tensor(rng.standard_normal((n_iters, x.shape[1])) * math.sqrt(local_radius),
                         dtype=torch.float32, device=x.device)
    return _grad_of_class(cm, x + noise, c).mean(0)


def m_integrated_gradient(cm, x_raw, c, baseline=None, num_iters=20, **_):
    """Their IG: `(x - b) * mean_alpha grad f(b + alpha (x - b))`, 20 steps, in
    STANDARDIZED space (their `baseline` args are `np.zeros` and the dataset min)."""
    x = cm.to_std(x_raw)
    b = torch.tensor(baseline, dtype=torch.float32, device=x.device).reshape(1, -1)
    alphas = torch.linspace(0, 1, num_iters, device=x.device).unsqueeze(1)
    g = _grad_of_class(cm, b + alphas * (x - b), c).mean(0)
    return (x - b).cpu().numpy()[0] * g


def m_lime(cm, x_raw, c, n_iters=100, local_radius=0.1, reg_param=1.0, rng=None, **_):
    """Port of their `lime`, including the single-row L2 regularization trick."""
    x = cm.to_std(x_raw).cpu().numpy()[0]
    feats = math.sqrt(local_radius) * rng.standard_normal((n_iters, x.shape[0])) + x
    with torch.no_grad():
        y = cm.torch_std(torch.tensor(feats, dtype=torch.float32,
                                      device=cm.device)).cpu().numpy()[:, c]
    A = np.vstack((feats, reg_param * np.ones((1, x.shape[0]))))
    b = np.hstack((y, [0.0]))
    return np.linalg.lstsq(A, b, rcond=None)[0]


def m_kernel_shap(cm, x_raw, c, background=None, nsamples=500, rng=None, **_):
    """KernelSHAP, reimplemented at their configuration (100 background rows sampled from
    the data, 500 coalition draws). Coalition sizes are drawn from the Shapley kernel
    (p ∝ (p-1)/(s(p-s))) and then uniformly within size, which makes the kernel weights
    constant and the regression an ordinary least squares -- the standard sampler. The
    efficiency constraint sum(phi) = f(x) - E[f] is imposed by eliminating one coordinate."""
    x = cm.to_std(x_raw).cpu().numpy()[0]
    bg = cm.to_std(background).cpu().numpy()
    p, nb = x.shape[0], bg.shape[0]

    def batched(rows):
        with torch.no_grad():
            return cm.torch_std(torch.tensor(rows, dtype=torch.float32,
                                             device=cm.device)).cpu().numpy()[:, c]

    phi0 = batched(bg).mean()
    fx = batched(x.reshape(1, -1))[0]

    sizes = np.arange(1, p)
    w = (p - 1) / (sizes * (p - sizes))
    s_draw = rng.choice(sizes, size=nsamples, p=w / w.sum())
    Z = np.zeros((nsamples, p))
    for i, s in enumerate(s_draw):
        Z[i, rng.choice(p, size=s, replace=False)] = 1.0

    rows = np.repeat(bg[None, :, :], nsamples, axis=0)             # [n, nb, p]
    rows[:, :, :] = np.where(Z[:, None, :] == 1, x[None, None, :], rows)
    y = batched(rows.reshape(-1, p)).reshape(nsamples, nb).mean(1) - phi0

    # eliminate phi_{p-1} via sum(phi) = fx - phi0
    delta = fx - phi0
    A = Z[:, :-1] - Z[:, -1:]
    b = y - delta * Z[:, -1]
    A = np.vstack([A, 1e-6 * np.eye(p - 1)])
    b = np.hstack([b, np.zeros(p - 1)])
    head = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.append(head, delta - head.sum())


def m_random(cm, x_raw, c, rng=None, **_):
    return rng.standard_normal(x_raw.shape[-1])

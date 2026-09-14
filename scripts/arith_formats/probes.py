"""Supervised probes + confidence baselines for the Fig. 4 / Sec. 3.4 comparison.

Paper protocol: logistic-regression probes on the last-prompt-token residual stream and on the
MLP activations (same units as the attribution), trained on 2,000 NUMERIC items held out from
the main set (1,600 correct / 400 incorrect), layer and C chosen by stratified 5-fold CV; the
probes are then applied to the main items in every format.  Outputs
``results/arith_formats/<model>/probes_<fmt>.json`` with per-item probe probabilities, which
``analyse.py`` enters into the linear probability model together with circuit loading, mean
log-prob and entropy.

Run after run_model.py --stage behav for the model (it reuses the Runner).
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import FORMATS, is_correct, make_items, prompt  # noqa: E402
from run_model import MODELS, Runner  # noqa: E402

log = logging.getLogger("probes")


@torch.no_grad()
def features(R, prompts, bs, layers_resid, layers_mlp):
    """Residual-stream (hidden_states) and MLP (down_proj input) features at the last prompt
    token for the requested layers.  Returns dict layer -> (N, dim) float16 numpy."""
    res = {("resid", l): [] for l in layers_resid}
    res.update({("mlp", l): [] for l in layers_mlp})
    for i in range(0, len(prompts), bs):
        P = prompts[i:i + bs]
        ids, attn, lp, *_ = R.batch(P, [[R.tok.eos_token_id]] * len(P))
        R.hook.set_rows(lp); R.hook.mode = "capture"; R.hook.acts = {}
        out = R.hf(input_ids=ids, attention_mask=attn, output_hidden_states=True)
        R.hook.mode = None
        ar = torch.arange(len(P), device=R.device)
        for l in layers_resid:
            res[("resid", l)].append(out.hidden_states[l][ar, lp].float().cpu().numpy().astype(np.float16))
        for l in layers_mlp:
            res[("mlp", l)].append(R.hook.acts[l].float().cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in res.items()}


def _cv_auc(X, y, C, skf):
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=500).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs))


def cv_select(Xs, y, Cs, seed):
    """Stratified 5-fold CV: layer search at the middle C, then a C sweep on the best layer
    (a full layer x C grid on 14k-dim MLP features takes hours). Returns best key, C, cv AUC."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    C0 = Cs[len(Cs) // 2]
    best = None
    for key, X in Xs.items():
        auc = _cv_auc(X.astype(np.float32), y, C0, skf)
        log.info("  %s C=%g  cv AUC=%.3f", key, C0, auc)
        if best is None or auc > best[2]:
            best = (key, C0, auc)
    key = best[0]
    for C in Cs:
        if C == C0:
            continue
        auc = _cv_auc(Xs[key].astype(np.float32), y, C, skf)
        log.info("  %s C=%g  cv AUC=%.3f", key, C, auc)
        if auc > best[2]:
            best = (key, C, auc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", default="results/arith_formats")
    ap.add_argument("--n-items", type=int, default=2000)
    ap.add_argument("--items-seed", type=int, default=0)
    ap.add_argument("--pool", type=int, default=8000, help="numeric pool to draw the probe set from")
    ap.add_argument("--pool-seed", type=int, default=1)
    ap.add_argument("--n-correct", type=int, default=1600)
    ap.add_argument("--n-incorrect", type=int, default=400)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--gen-bs", type=int, default=64)
    ap.add_argument("--layer-step", type=int, default=2)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = Path(args.out) / args.model
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R = Runner(args.model, device, out)
    main_items = json.load(open(Path(args.out) / f"items_{args.n_items}_{args.items_seed}.json"))
    main_keys = {(tuple(it["operands"]), tuple(it["ops"])) for it in main_items}

    # ---- probe training set: numeric items disjoint from the main set, 80/20 correct/incorrect
    fn_pool = out / f"probe_pool_{args.pool}.json"
    if fn_pool.exists():
        pool = json.load(open(fn_pool))
    else:
        pool = [it for it in make_items(args.pool, args.pool_seed)
                if (tuple(it["operands"]), tuple(it["ops"])) not in main_keys]
        P = [prompt(it, "numeric") for it in pool]
        for i in range(0, len(pool), args.gen_bs):
            for it, (ids, text) in zip(pool[i:i + args.gen_bs], R.generate(P[i:i + args.gen_bs])):
                it["gen"] = text; it["correct"] = is_correct(text, it, "numeric")
        json.dump(pool, open(fn_pool, "w"))
    rng = random.Random(args.seed)
    corr = [it for it in pool if it["correct"]]; inc = [it for it in pool if not it["correct"]]
    log.info("pool: %d correct, %d incorrect (acc %.3f)", len(corr), len(inc), len(corr) / len(pool))
    n_inc = min(args.n_incorrect, len(inc)); n_cor = min(args.n_correct, len(corr))
    train = rng.sample(corr, n_cor) + rng.sample(inc, n_inc)
    rng.shuffle(train)
    y = np.array([int(it["correct"]) for it in train])
    log.info("probe train set: %d items (%d correct / %d incorrect)", len(train), n_cor, n_inc)

    L = R.L
    layers_resid = list(range(1, L + 1, args.layer_step))      # hidden_states[l] = after block l-1
    layers_mlp = list(range(0, L, args.layer_step))
    Ptr = [prompt(it, "numeric") for it in train]
    feats = features(R, Ptr, args.bs, layers_resid, layers_mlp)
    log.info("selecting residual probe")
    best_r = cv_select({k: v for k, v in feats.items() if k[0] == "resid"}, y, args.Cs, args.seed)
    log.info("selecting MLP probe")
    best_m = cv_select({k: v for k, v in feats.items() if k[0] == "mlp"}, y, args.Cs, args.seed)
    log.info("best resid %s  best mlp %s", best_r, best_m)
    probes = {}
    for best in (best_r, best_m):
        key, C, auc = best
        X = feats[key].astype(np.float32)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=C, max_iter=5000).fit(sc.transform(X), y)
        probes[key[0]] = (key[1], C, auc, sc, clf)
    del feats

    # ---- apply to the main items in every format
    for fmt in FORMATS:
        P = [prompt(it, fmt) for it in main_items]
        f = features(R, P, args.bs, [probes["resid"][0]], [probes["mlp"][0]])
        rec = {"format": fmt, "layer": {k: probes[k][0] for k in probes},
               "C": {k: probes[k][1] for k in probes}, "cv_auc": {k: round(probes[k][2], 4) for k in probes},
               "n_train": len(train), "resid": {}, "mlp": {}}
        for k in ("resid", "mlp"):
            layer, C, auc, sc, clf = probes[k]
            p = clf.predict_proba(sc.transform(f[(k, layer)].astype(np.float32)))[:, 1]
            for it, pp in zip(main_items, p):
                rec[k][str(it["id"])] = float(pp)
        # in-format AUC of the numeric-trained probe, for the record
        beh = out / f"behav_{fmt}.json"
        if beh.exists():
            cm = {r["id"]: r["correct"] for r in json.load(open(beh))["items"]}
            yy = np.array([cm[it["id"]] for it in main_items])
            if 0 < yy.mean() < 1:
                rec["auc_on_format"] = {k: float(roc_auc_score(yy, [rec[k][str(it["id"])] for it in main_items]))
                                        for k in ("resid", "mlp")}
                log.info("[%s] probe AUC on this format: %s", fmt, rec["auc_on_format"])
        json.dump(rec, open(out / f"probes_{fmt}.json", "w"))


if __name__ == "__main__":
    main()

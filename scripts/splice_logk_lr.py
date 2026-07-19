"""Splice the log-k block (METHODS[0]) into paper/tabs/lr_sweep.tex without touching the
uniform-k / REINFORCE blocks (whose result dirs live on sc, not here). Reuses make_lr_table's
cpr/fmt/DIR_OVERRIDE/DAGGER logic. Idempotent: replaces any existing log-k block.
Run from repo root: .venv/bin/python scripts/splice_logk_lr.py
"""
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
import make_lr_table as M

TEX = Path("paper/tabs/lr_sweep.tex")
ncols = len(M.COLUMNS)


def logk_block(method, lrs):
    data = {lr: {(t, m): M.cpr(d, t, m) for t, m, _ in M.COLUMNS} for lr, d in lrs}
    best = {}
    for t, m, _ in M.COLUMNS:
        vals = [data[lr][(t, m)] for lr, _ in lrs if data[lr][(t, m)] is not None]
        best[(t, m)] = max(vals) if len(vals) > 1 else None
    out = [f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{{method}}}}} \\\\"]
    for lr, _ in lrs:
        present = [data[lr][(t, m)] for t, m, _ in M.COLUMNS if data[lr][(t, m)] is not None]
        avg = f"{sum(present) / len(present):.2f}" if present else "---"
        cells = [M.fmt(data[lr][(t, m)],
                       bold=(data[lr][(t, m)] is not None and data[lr][(t, m)] == best[(t, m)]),
                       dagger=((t, m) in M.DAGGER_CELLS and data[lr][(t, m)] is not None))
                 for t, m, _ in M.COLUMNS]
        out.append(f"\\quad LR$=${lr} & {avg} & " + " & ".join(cells) + " \\\\")
    return out


lines = TEX.read_text().splitlines()
# drop any previously spliced log-k block (marker: the log-k \multicolumn line + its rows + a \midrule)
cleaned, skip = [], False
for ln in lines:
    if "log $k$" in ln and "multicolumn" in ln:
        skip = True
        continue
    if skip:
        if ln.strip() == "\\midrule":   # end of the old block
            skip = False
        continue
    cleaned.append(ln)

# all log-k METHODS blocks (hard fwd, soft fwd, ...) — insert after the first \midrule,
# before the uniform-k block, each followed by a \midrule.
logk_methods = [(mth, lrs) for mth, lrs in M.METHODS if "log $k$" in mth]
blocks = []
for mth, lrs in logk_methods:
    blocks += logk_block(mth, lrs) + ["\\midrule"]

out, done = [], False
for ln in cleaned:
    out.append(ln)
    if not done and ln.strip() == "\\midrule":
        out += blocks
        done = True
TEX.write_text("\n".join(out) + "\n")
print(f"spliced {len(logk_methods)} log-k block(s) into {TEX}")
for mth, lrs in logk_methods:
    print("\n".join(logk_block(mth, lrs)))

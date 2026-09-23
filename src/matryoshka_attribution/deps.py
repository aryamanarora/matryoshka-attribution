"""Locate the outside repos this project calls unmodified.

MIB-circuit-track is OUR FORK (github.com/aryamanarora/MIB-circuit-track, with the EAP-IG
submodule), cloned by `scripts/setup.sh` into `deps/MIB-circuit-track` at a pinned commit and
gitignored there. Upstream MIB's `evaluate_area_under_curve` returns 5 values; the fork's returns
7 (`accuracies`, `acc_auc`) and every `scripts/mib/eval_*.py` unpacks 7, so an upstream or stale
clone trains for hours and then dies AFTER eval, before `scores.pt` is written.

Resolution order, first hit wins:
  1. an explicit path (`--mib-path` / the `explicit` argument)
  2. `$MATTR_MIB_PATH`
  3. `<repo>/deps/MIB-circuit-track`          (scripts/setup.sh's layout)
  4. `<repo>/MIB-circuit-track`               (an older gitignored symlink, if one exists)
  5. `./MIB-circuit-track` relative to the CWD (the scripts' historical default)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIB_NAME = "MIB-circuit-track"
_MARKER = "run_evaluation.py"


def _looks_like_mib(p: Path) -> bool:
    return (p / _MARKER).is_file() and (p / "EAP-IG" / "src" / "eap").is_dir()


def find_mib_path(explicit: str | os.PathLike | None = None) -> Path:
    """Return the MIB-circuit-track checkout to use (resolved), or raise with the fix."""
    candidates = []
    if explicit:
        candidates.append(("--mib-path", Path(explicit)))
    if os.environ.get("MATTR_MIB_PATH"):
        candidates.append(("$MATTR_MIB_PATH", Path(os.environ["MATTR_MIB_PATH"])))
    candidates += [
        ("deps/", REPO_ROOT / "deps" / MIB_NAME),
        ("legacy symlink", REPO_ROOT / MIB_NAME),
        ("cwd", Path(MIB_NAME)),
    ]
    for how, p in candidates:
        if _looks_like_mib(p):
            return p.resolve()
        if how in ("--mib-path", "$MATTR_MIB_PATH"):
            raise FileNotFoundError(
                f"{how}={p} is not a MIB-circuit-track checkout with its EAP-IG submodule "
                f"(need {_MARKER} and EAP-IG/src/eap). Clone with --recurse-submodules.")
    raise FileNotFoundError(
        "MIB-circuit-track not found. Run `bash scripts/setup.sh` (clones our fork with its "
        "EAP-IG submodule into deps/MIB-circuit-track at the pinned commit), or pass --mib-path "
        "/ set $MATTR_MIB_PATH. Looked in: " + ", ".join(str(p) for _, p in candidates))


def mib_results_dir(explicit: str | os.PathLike | None = None) -> Path:
    """The MIB fork's own `results/` tree (run_attribution / run_evaluation outputs, the
    `*_accauc` re-eval mirrors, `mattr_accauc*`), which the table generators read baseline
    rows from. Resolved through find_mib_path()."""
    return find_mib_path(explicit) / "results"


def add_mib_to_sys_path(explicit: str | os.PathLike | None = None) -> Path:
    """find_mib_path() + the two sys.path entries every MIB-calling script needs."""
    p = find_mib_path(explicit)
    sys.path.insert(0, str(p))
    sys.path.insert(0, str(p / "EAP-IG" / "src"))
    return p

"""Locate (and if needed fetch) the MIB benchmark fork this project calls unmodified.

MIB-circuit-track is OUR FORK (github.com/aryamanarora/MIB-circuit-track, with the EAP-IG
submodule), kept at `deps/MIB-circuit-track` (gitignored) at the commit pinned below. Upstream
MIB's `evaluate_area_under_curve` returns 5 values; the fork's returns 7 (`accuracies`,
`acc_auc`) and every `scripts/mib/eval_*.py` unpacks 7, so an upstream or stale clone trains
for hours and then dies AFTER eval, before `scores.pt` is written. The pin is what every number
in results/ was produced against; move it in the same commit as the numbers it changes.

Resolution order, first hit wins:
  1. an explicit path (`--mib-path` / the `explicit` argument)
  2. `$MATTR_MIB_PATH`
  3. `<repo>/deps/MIB-circuit-track`, cloned at the pin on first use if absent
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIB_NAME = "MIB-circuit-track"
MIB_URL = "https://github.com/aryamanarora/MIB-circuit-track.git"
# aryamanarora/MIB-circuit-track main, 2026-09-18: EAP-IG @ 3fb341e (AtP / AtP-star node methods,
# None-grad fix); before that d7c76bd (EAP-IG c2dd06c, intervention=zero for EAP-IG-inputs / -mc /
# AttnLRP) and 2324d9a (invert / extra[flip_accuracies]).
MIB_SHA = "582fdbc"
_MARKER = "run_evaluation.py"


def _looks_like_mib(p: Path) -> bool:
    return (p / _MARKER).is_file() and (p / "EAP-IG" / "src" / "eap").is_dir()


def _clone(dest: Path) -> None:
    """Clone the fork with its submodule into ``dest`` and check out the pin."""
    url = os.environ.get("MATTR_MIB_URL", MIB_URL)
    print(f"[matryoshka_attribution.deps] cloning {url} -> {dest} @ {MIB_SHA}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--recurse-submodules", url, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", MIB_SHA], check=True)
    subprocess.run(["git", "-C", str(dest), "submodule", "update", "--init", "--quiet"], check=True)


def find_mib_path(explicit: str | os.PathLike | None = None, clone: bool = True) -> Path:
    """Return the MIB-circuit-track checkout to use (resolved), cloning it if needed."""
    for how, p in ((("--mib-path", Path(explicit)),) if explicit else ()) + \
                  ((("$MATTR_MIB_PATH", Path(os.environ["MATTR_MIB_PATH"])),)
                   if os.environ.get("MATTR_MIB_PATH") else ()):
        if _looks_like_mib(p):
            return p.resolve()
        raise FileNotFoundError(
            f"{how}={p} is not a MIB-circuit-track checkout with its EAP-IG submodule "
            f"(need {_MARKER} and EAP-IG/src/eap). Clone with --recurse-submodules.")
    dest = REPO_ROOT / "deps" / MIB_NAME
    if not _looks_like_mib(dest):
        if not clone:
            raise FileNotFoundError(f"{dest} is not a MIB-circuit-track checkout; run scripts/setup.sh")
        _clone(dest)
        assert _looks_like_mib(dest), f"clone of {MIB_URL} at {dest} lacks {_MARKER} or the EAP-IG submodule"
    return dest.resolve()


def mib_results_dir(explicit: str | os.PathLike | None = None) -> Path:
    """The MIB fork's own `results/` tree (run_attribution / run_evaluation outputs, the
    `*_accauc` re-eval mirrors, `mattr_accauc*`), which the table generators read baseline
    rows from."""
    return find_mib_path(explicit) / "results"


def add_mib_to_sys_path(explicit: str | os.PathLike | None = None) -> Path:
    """find_mib_path() + the two sys.path entries every MIB-calling script needs."""
    p = find_mib_path(explicit)
    sys.path.insert(0, str(p))
    sys.path.insert(0, str(p / "EAP-IG" / "src"))
    return p

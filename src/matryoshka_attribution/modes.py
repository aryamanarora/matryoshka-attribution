"""Intervention direction names.

  - ``iso``   ("isolation"): the top-k stays CLEAN and the complement is patched to the source
    (denoising). Recovers clean behaviour; what MIB's CPR measures and what every MAttr run uses.
  - ``cause`` ("causation"): the top-k is patched to the source and the complement stays clean
    (noising). Finds what breaks behaviour.

``sufficient`` / ``necessary`` were the names until 2026-09-22; ``normalize_mode`` folds them
onto the current ones so records written before then still read.
"""

ISO, CAUSE = "iso", "cause"
MODE_CHOICES = [ISO, CAUSE]

_LEGACY = {"sufficient": ISO, "necessary": CAUSE}


def normalize_mode(mode: str) -> str:
    """``iso``/``cause`` (or a legacy ``sufficient``/``necessary``) -> ``iso``/``cause``."""
    mode = _LEGACY.get(mode, mode)
    if mode not in MODE_CHOICES:
        raise ValueError(f"unknown intervention mode {mode!r}; use {MODE_CHOICES}")
    return mode

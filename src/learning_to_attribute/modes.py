"""Intervention-type naming: the new preferred names ``iso`` / ``cause`` and their mapping
to the canonical internal names ``sufficient`` / ``necessary``.

Vocabulary (see CLAUDE.md for the load-bearing convention):
  - ``iso``   ("isolation")  == ``sufficient`` == denoising: top-k stays CLEAN, complement
    corrupted; recover clean behaviour (our MIB / MAttr runs).
  - ``cause`` ("causation")  == ``necessary``  == noising: top-k corrupted, complement clean;
    find what breaks behaviour.

Both name sets are accepted everywhere (CLI flags, config keys, env vars) during the
transition; entry points call :func:`normalize_mode` to fold the new names onto the existing
canonical values, so all downstream logic, saved configs, and result-dir names are unchanged.
"""

# canonical -> preferred alias, and the reverse, plus identity mappings
ISO, CAUSE = "iso", "cause"
SUFFICIENT, NECESSARY = "sufficient", "necessary"

_TO_CANONICAL = {
    "iso": SUFFICIENT, "sufficient": SUFFICIENT,
    "cause": NECESSARY, "necessary": NECESSARY,
}
_TO_PREFERRED = {SUFFICIENT: ISO, NECESSARY: CAUSE}

# choices lists for argparse: new names first (preferred), old kept for back-compat
MODE_CHOICES = [ISO, CAUSE, SUFFICIENT, NECESSARY]


def normalize_mode(mode: str) -> str:
    """Fold any accepted intervention-type name to canonical ``{'sufficient','necessary'}``.

    Accepts ``iso``/``cause`` (preferred) or ``sufficient``/``necessary`` (legacy).
    """
    try:
        return _TO_CANONICAL[mode]
    except KeyError:
        raise ValueError(
            f"unknown intervention mode {mode!r}; use {MODE_CHOICES}") from None


def preferred_mode(mode: str) -> str:
    """Return the preferred (iso/cause) name for any accepted mode string."""
    return _TO_PREFERRED[normalize_mode(mode)]

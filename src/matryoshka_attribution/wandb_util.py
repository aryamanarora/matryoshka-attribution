"""One wandb entry point for every trainer in the repo (eval_sva, eval_mib, eval_mib_edge,
attribute).

Two conventions this module exists to enforce:

**One project per dataset** (``l2a-sva``, ``l2a-arith``, ``l2a-causalgym``, ``l2a-mib``).
Runs are only comparable within a dataset: the substrates differ (2.3M MLP neurons vs 1056
MIB nodes), the unit counts differ by three orders of magnitude, and so do the metric scales.
A single project would overlay incommensurable series and its run table would not sort.

**Logging is never fatal.** These are multi-hour GPU jobs on a cluster where some nodes have
no credentials; a wandb client that cannot reach the API, or that renames a property in a
point release, must not take a training run down with it. Everything here degrades: no
credentials -> offline (recoverable later with ``wandb sync``), any other failure -> ``None``,
which every call site already treats as "logging off".
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

#: dataset -> wandb project. Anything not listed falls back to ``l2a-<dataset>``.
PROJECTS = {"sva": "l2a-sva", "arith": "l2a-arith", "causalgym": "l2a-causalgym",
            "mib": "l2a-mib"}


def project_for(dataset):
    return PROJECTS.get(dataset, f"l2a-{dataset}")


def init(dataset, name, config, *, project=None, entity=None, enabled=True, **kw):
    """Start a run, or return None. ``config`` is filtered to json-safe scalars."""
    if not enabled:
        return None
    try:
        import wandb
        # No credentials on some nodes; fall back to offline rather than losing the run.
        # `wandb sync <dir>` uploads it once a key is available.
        if not (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
            os.environ.setdefault("WANDB_MODE", "offline")
            logger.warning("no WANDB_API_KEY and no ~/.netrc -> logging OFFLINE")
        cfg = {k: v for k, v in dict(config).items()
               if isinstance(v, (int, float, str, bool, type(None)))}
        return wandb.init(entity=entity, project=project or project_for(dataset),
                          name=name, config=cfg, **kw)
    except Exception as exc:                       # noqa: BLE001 -- logging must never be fatal
        logger.warning("wandb disabled (%s: %s)", type(exc).__name__, exc)
        return None


def add_args(parser, *, dash=True):
    """Add ``--no-wandb`` / ``--wandb-project`` / ``--wandb-entity``.

    ON BY DEFAULT: these runs are hours long and their only other record is a slurm .err file
    that nobody diffs across 400 jobs. ``dash=False`` spells the two option names with
    underscores, for the scripts (attribute.py) that already use that style.
    """
    sep = "-" if dash else "_"
    parser.add_argument("--no-wandb", dest="wandb", action="store_false",
                        help="disable wandb logging (on by default)")
    parser.add_argument(f"--wandb{sep}project", default=None,
                        help="override the per-dataset default project (see wandb_util.PROJECTS)")
    parser.add_argument(f"--wandb{sep}entity", default=None, help="wandb entity")
    return parser

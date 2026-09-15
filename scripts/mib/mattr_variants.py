"""ONE definition of which MAttr variant is the unmarked headline row, and how every other
variant is labelled relative to it. Imported by the MIB tables (make_mib_table.py,
make_mib_accauc_table.py, make_mib_test_table.py), the SVA tables (scripts/sva/make_sva_table.py)
and the plots that key on those row strings, so a change of headline is one edit here.

HEADLINE (user decision 2026-09-15): SOFT top-k forward, UNIFORM k-schedule, ADAM. On the SVA
substrates, where Adam's epsilon is swept, the headline is eps = 1e-2 and the default-eps run
(1e-8) is the annotated variant.

History, so nobody reads an old table against a new one:
  2026-06-17  uniform k, Adam            (mib_node_hard_topk era)
  2026-07-21  soft fwd, LOG k, Adam      (topklog_lr_0.05)
  2026-08-24  soft fwd, log k, SGD       (softlog_sgd_lr_1.0; edge lr 3.0 from 2026-09-04)
  2026-09-15  soft fwd, UNIFORM k, Adam  (mib_node_topk_uniform_lr05 / test_node_topk_uniform_lr05;
                                          edge mib_edge_topk_uniform_lr05 / test_edge_topk_uniform_lr05)
The bar chart (plots/plot_mib_test_avg.py) had drawn the uniform-k Adam run as the plain method
since 2026-09-11; this makes the tables agree with it.

Labelling grammar (comma-separated, fixed order k-schedule, optimizer, forward, eps, extras):
  label()                       -> \\ourmethod{}
  label(k="log")                -> $+$ log $k$
  label(k="log", opt="sgd")     -> $+$ log $k$, $+$ SGD
  label(opt="sgd", fwd="hard")  -> $+$ SGD, $+$ hard
  label(k="log", eps="1e-8")    -> $+$ log $k$, $\\epsilon{=}10^{-8}$
An attribute equal to the headline's contributes nothing, so the same call sites survive the
next headline change untouched.
"""

HEADLINE = {"k": "uniform", "opt": "adam", "fwd": "soft"}
SVA_HEADLINE_EPS = "1e-2"    # Adam epsilon; only swept (and only labelled) on the SVA substrates

OURMETHOD = "\\ourmethod{}"
K_MARK = {"log": "$+$ log $k$", "uniform": "$+$ unif $k$"}
OPT_MARK = {"sgd": "$+$ SGD", "adam": "$+$ Adam"}
FWD_MARK = {"hard": "$+$ hard", "soft": "$+$ soft"}
OPT_NAME = {"sgd": "SGD", "adam": "Adam"}

# make_mib_table.OUR_METHODS files rows under two k-schedule groups; this is the map from its
# group key to the schedule it means.
GROUP_K = {"ours": "log", "uniform": "uniform"}


def eps_mark(eps):
    """'1e-8' -> $\\epsilon{=}10^{-8}$ (LaTeX, math mode)."""
    mant, exp = str(eps).lower().split("e")
    exp = int(exp)
    return (f"$\\epsilon{{=}}10^{{{exp}}}$" if float(mant) == 1.0
            else f"$\\epsilon{{=}}{mant}{{\\times}}10^{{{exp}}}$")


def marks(k=None, opt=None, fwd=None, eps=None, extra=()):
    """The list of ablation marks for a variant, empty for the headline."""
    out = []
    if k is not None and k != HEADLINE["k"]:
        out.append(K_MARK[k])
    if opt is not None and opt != HEADLINE["opt"]:
        out.append(OPT_MARK[opt])
    if fwd is not None and fwd != HEADLINE["fwd"]:
        out.append(FWD_MARK[fwd])
    if eps is not None and str(eps).lower() != SVA_HEADLINE_EPS:
        out.append(eps_mark(eps))
    out += list(extra)
    return out


def label(k=None, opt=None, fwd=None, eps=None, extra=()):
    """Row label for a variant: bare \\ourmethod{} for the headline, else its marks joined."""
    m = marks(k, opt, fwd, eps, extra)
    return OURMETHOD if not m else ", ".join(m)


def mark_k(name, k):
    """Prefix an existing relative row label (e.g. "$+$ hard", or bare \\ourmethod{}) with the
    k-schedule mark when `k` is not the headline schedule. Replaces the old per-table `unifk()`."""
    if k == HEADLINE["k"]:
        return name
    return K_MARK[k] if name == OURMETHOD else f"{K_MARK[k]}, {name}"


def opt_header(opt):
    """Block header used by the validation tables: \\ourmethod{} for the headline optimizer,
    \\ourmethod{}$+$SGD (etc.) for the other."""
    return OURMETHOD if opt == HEADLINE["opt"] else f"{OURMETHOD}$+${OPT_NAME[opt]}"


# Emission order for the optimizer blocks: headline optimizer first and unmarked.
OPT_ORDER = [(HEADLINE["opt"], opt_header(HEADLINE["opt"]))] + \
            [(o, opt_header(o)) for o in ("adam", "sgd") if o != HEADLINE["opt"]]
# Emission order for the k-schedule groups of make_mib_table.OUR_METHODS: headline group first.
GROUP_ORDER = sorted(GROUP_K, key=lambda g: GROUP_K[g] != HEADLINE["k"])

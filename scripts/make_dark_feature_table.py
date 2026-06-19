"""LaTeX longtable: one row per unique (layer,dim) dark-red DAS feature, with the number
of tasks it tops + a coloured category tag per category it appears in, and its top-5 /
bottom-5 logit-lens tokens. Writes paper/tabs/dark_feature_logitlens.tex.
Needs in the preamble: \\usepackage{tikz}, \\usepackage{longtable}, \\methodtag + cat tags."""
import json
from collections import defaultdict

d = json.load(open("results/mtdas_dark_feature_logitlens.json"))

CATS = ["Agreement", "Licensing", "Garden", "GSS", "Long"]
TAG = {"Agreement": r"\agtag", "Licensing": r"\litag", "Garden": r"\gatag",
       "GSS": r"\gstag", "Long": r"\lotag"}
CAT = {}
for t in ["agr_gender", "agr_sv_num_subj-relc", "agr_sv_num_obj-relc", "agr_sv_num_pp",
          "agr_refl_num_subj-relc", "agr_refl_num_obj-relc", "agr_refl_num_pp"]: CAT[t] = "Agreement"
for t in ["npi_any_subj-relc", "npi_any_obj-relc", "npi_ever_subj-relc", "npi_ever_obj-relc"]: CAT[t] = "Licensing"
for t in ["garden_mvrr", "garden_mvrr_mod", "garden_npz_obj", "garden_npz_obj_mod",
          "garden_npz_v-trans", "garden_npz_v-trans_mod"]: CAT[t] = "Garden"
for t in ["gss_subord", "gss_subord_subj-relc", "gss_subord_obj-relc", "gss_subord_pp"]: CAT[t] = "GSS"
for t in ["cleft", "cleft_mod", "filler_gap_embed_3", "filler_gap_embed_4",
          "filler_gap_hierarchy", "filler_gap_obj", "filler_gap_pp", "filler_gap_subj"]: CAT[t] = "Long"

SPECIAL = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
           ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
           ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def esc(s):
    for a, b in SPECIAL:
        s = s.replace(a, b)
    return s


def tokfmt(toks):
    out = []
    for t in toks:
        s = "".join(c if 32 <= ord(c) <= 126 else "?" for c in t)  # ASCII-safe
        s = s.replace(" ", "_")          # visible space marker (GPT-style)
        if len(s) > 14:
            s = s[:13] + ".."
        out.append(r"\toktag{" + esc(s) + "}")   # little cell per token
    return " ".join(out)


feat = defaultdict(lambda: {"tasks": [], "top": None, "bot": None})
for task, entries in d.items():
    for e in entries:
        k = (e["layer"], e["dim"])
        feat[k]["tasks"].append(task)
        feat[k]["top"] = e["top5"]
        feat[k]["bot"] = e["bottom5"]

rows = sorted(feat.items(), key=lambda kv: (kv[0][0], -len(kv[1]["tasks"]), kv[0][1]))

L = [r"{\small",
     r"\begin{longtable}{@{}r r r l p{3.6cm} p{3.6cm}@{}}",
     r"\toprule",
     r"Layer & Dim & \# & Cat. & Top-5 logits & Bottom-5 logits \\",
     r"\midrule \endhead"]
for i, ((lyr, dim), v) in enumerate(rows):
    cats = [c for c in CATS if c in {CAT[t] for t in v["tasks"]}]
    tags = "".join(TAG[c] for c in cats)
    shade = r"\rowcolor{rowgray}" if i % 2 else ""
    L.append("%s%d & %d & %d & %s & %s & %s \\\\" % (
        shade, lyr, dim, len(v["tasks"]), tags, tokfmt(v["top"]), tokfmt(v["bot"])))
L += [r"\bottomrule", r"\end{longtable}", r"}"]
open("paper/tabs/dark_feature_logitlens.tex", "w").write("\n".join(L) + "\n")
print("rows:", len(rows), "| multi-task:", sum(1 for _, v in rows if len(v["tasks"]) > 1))

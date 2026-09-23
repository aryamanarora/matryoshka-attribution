"""One-off migration of result files from the pre-2026-09-22 intervention names.

Renames every file under --root whose name carries a `_sufficient_` / `_necessary_` tag segment
to `_iso_` / `_cause_`, and rewrites the saved `mode` field inside eval_sva jsons (top level and
`config.mode`) and inside eval_mib `*_scores.pt` args dicts. Dry run by default; --apply writes.

    uv run python scripts/tools/migrate_mode_names.py --root results            # list
    uv run python scripts/tools/migrate_mode_names.py --root results --apply    # do it
"""
import argparse
import json
from pathlib import Path

RENAME = {"sufficient": "iso", "necessary": "cause"}


def new_name(name):
    for old, new in RENAME.items():
        name = name.replace(f"_{old}_", f"_{new}_")
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    renames, json_fixes, pt_fixes = [], [], []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix == ".json":
            try:
                d = json.load(open(p))
            except Exception:
                d = None
            if isinstance(d, dict):
                m = d.get("mode"); c = (d.get("config") or {}).get("mode") if isinstance(d.get("config"), dict) else None
                if m in RENAME or c in RENAME:
                    json_fixes.append(p)
        elif p.name.endswith("_scores.pt"):
            pt_fixes.append(p)          # checked lazily (needs torch)
        if new_name(p.name) != p.name:
            renames.append(p)
    print(f"{len(renames)} files to rename, {len(json_fixes)} jsons with a legacy mode field, "
          f"{len(pt_fixes)} scores.pt to inspect")
    for p in renames[:40]:
        print("  rename", p, "->", new_name(p.name))
    for p in json_fixes[:20]:
        print("  json  ", p)
    if not args.apply:
        return
    for p in json_fixes:
        d = json.load(open(p))
        if d.get("mode") in RENAME:
            d["mode"] = RENAME[d["mode"]]
        if isinstance(d.get("config"), dict) and d["config"].get("mode") in RENAME:
            d["config"]["mode"] = RENAME[d["config"]["mode"]]
        json.dump(d, open(p, "w"), indent=2)
    import torch
    n_pt = 0
    for p in pt_fixes:
        d = torch.load(p, map_location="cpu", weights_only=False)
        a = d.get("args") if isinstance(d, dict) else None
        if isinstance(a, dict) and a.get("mode") in RENAME:
            a["mode"] = RENAME[a["mode"]]
            torch.save(d, p); n_pt += 1
    for p in renames:
        p.rename(p.with_name(new_name(p.name)))
    print(f"applied: {len(renames)} renamed, {len(json_fixes)} jsons rewritten, {n_pt} scores.pt rewritten")


if __name__ == "__main__":
    main()

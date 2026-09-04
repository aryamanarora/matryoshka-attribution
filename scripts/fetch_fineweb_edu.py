"""Fetch a small FineWeb-Edu sample to a local jsonl (one doc per line).

Pulls rows through the HF datasets-server `/rows` endpoint rather than
`load_dataset(..., streaming=True)`: streaming the `sample-10BT` config has to resolve and
open multi-GB parquet shards before it yields a single document (>5 min here for 2 docs),
whereas `/rows` returns a page of 100 in well under a second. Same rows, same order.

The point of writing to disk is that every run downstream reads the SAME documents, on a
cluster node with no network guarantee, and the sample is a git-visible artifact of the
experiment rather than whatever the endpoint served that day.

    python scripts/fetch_fineweb_edu.py --n 200 --out data/fineweb_edu_200.jsonl
"""
import argparse, json, time
from pathlib import Path

import requests

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100          # endpoint's max rows per request


def fetch(n, dataset, config, split, offset, retries=5):
    out, got = [], 0
    while got < n:
        want = min(PAGE, n - got)
        params = dict(dataset=dataset, config=config, split=split,
                      offset=offset + got, length=want)
        for attempt in range(retries):
            r = requests.get(ROWS_URL, params=params, timeout=60)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt)
        else:
            raise SystemExit(f"datasets-server failed: {r.status_code} {r.text[:200]}")
        rows = r.json()["rows"]
        if not rows:
            raise SystemExit(f"ran out of rows at offset {offset + got}")
        out.extend(row["row"] for row in rows)
        got += len(rows)
        print(f"  {got}/{n}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out = Path(a.out or f"data/fineweb_edu_{a.n}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = fetch(a.n, a.dataset, a.config, a.split, a.offset)
    keep = ("text", "id", "url", "token_count", "score", "language")
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keep if k in r}) + "\n")
    print(f"wrote {len(rows)} docs -> {out}")


if __name__ == "__main__":
    main()

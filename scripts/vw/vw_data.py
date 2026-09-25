"""Corpus + tokenizer for the 1L transformer of Turner, Wu & Batson, "Characterizing
interference weights in a tiny language model" (Transformer Circuits, 2026-08-21).

    uv run python scripts/vw/vw_data.py --seqs 95464          # ~98M train tokens + a val split

Everything here is quoted from that note's Appendix > Training details; nothing is a
free knob except how many parquet files we pull (a throughput detail, not a config):

  tokenizer   "a 4,096-token BPE vocabulary obtained by truncating the tokenizer of the
              publicly released Pleias-1.2B model to its first 4,096 token ids, keeping
              only the BPE merges whose inputs and output all survive the truncation."
  corpus      "the openly licensed Common Corpus (PleIAs), restricted to documents
              labeled as English or as code (the filter is a disjunction over the
              corpus's language/language_type metadata columns)."
  shards      "The corpus copy is split into ten file shards; nine are used for training
              and the tenth is held out for evaluation."  -- the HF repo is *literally*
              laid out as common_corpus_1..10, so we take that as the split.
  packing     "Each training sequence is 1,024 tokens: it begins with a sequence
              delimiter token, packs consecutive documents each prefixed by the sequence
              delimiter, and discards any remainder beyond 1,024 tokens (no carryover
              between sequences)."
  size        "≈9.8x10^7 unique tokens (95,464 unique sequences)".

VALIDATION THAT THE TOKENIZER IS THE RIGHT ONE, and it is a strong one: the note's worked
example is ACETYLCHOLINE, "predicting that the final ' E ' token follows ' IN '". Under
this truncation the string tokenizes as  AC ET Y L CH OL IN E  -- the ' IN '->' E ' pair
the note analyses falls out of the reconstruction rather than being assumed. `--check`
prints it. (13 of the 256 byte-level base tokens fall outside the first 4096 ids and are
lost; that is a property of the stated truncation, not a choice made here.)

Writes data/vw/{tok4096.json,train.bin,val.bin,meta.json}; the .bin files are flat uint16
arrays of 1024-token sequences (vocab 4096 fits in uint16 with room to spare).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "vw"
REPO = "PleIAs/common_corpus"
TOK_MODEL = "PleIAs/Pleias-1.2b-Preview"
VOCAB, CTX = 4096, 1024
DELIM = 1  # <|begin_of_text|> in the Pleias vocab; id 0 is [UNK], 2 is eos, 3 is pad


def build_tokenizer(path: Path):
    """Truncate the Pleias BPE to its first 4096 ids, keeping only surviving merges."""
    import json as _json

    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    src = _json.load(open(hf_hub_download(TOK_MODEL, "tokenizer.json")))
    vocab = {t: i for t, i in src["model"]["vocab"].items() if i < VOCAB}
    assert len(vocab) == VOCAB, len(vocab)
    # "keeping only the BPE merges whose inputs and output all survive the truncation"
    merges = [m for m in src["model"]["merges"]
              if m[0] in vocab and m[1] in vocab and (m[0] + m[1]) in vocab]
    out = _json.loads(_json.dumps(src))
    out["model"]["vocab"] = vocab
    out["model"]["merges"] = merges
    out["added_tokens"] = [t for t in src["added_tokens"] if t["id"] < VOCAB]
    path.write_text(_json.dumps(out))
    tk = Tokenizer.from_file(str(path))
    print(f"tokenizer: {tk.get_vocab_size()} ids, {len(merges)}/{len(src['model']['merges'])} merges kept")
    return tk


def keep(row):
    """The note's filter: 'documents labeled as English or as code', a disjunction."""
    return row.get("language") == "English" or row.get("language_type") == "Code"


def pack(tokenizer, texts, seqs, target, seen):
    """Fill `seqs` with 1024-token sequences; no carryover past a sequence boundary."""
    cur = [DELIM]
    for enc in tokenizer.encode_batch_fast(texts):
        cur.extend(enc.ids)
        while len(cur) >= CTX:
            seqs.append(np.asarray(cur[:CTX], dtype=np.uint16))
            cur = [DELIM]                      # discard the remainder, no carryover
            if len(seqs) >= target:
                return True
        cur.append(DELIM)                      # each document prefixed by the delimiter
    return False


def build_split(tokenizer, dirs, target, tag):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if any(f.startswith(d + "/") for d in dirs) and f.endswith(".parquet")]
    # Round-robin across the nine training shards rather than draining shard 1: the note
    # says "nine are used for training", and Common Corpus's shards are not interchangeable
    # (each is a different slice of sources), so taking all the data from one would train on
    # a narrower distribution than the note's.
    per = {}
    for f in files:
        per.setdefault(f.split("/")[0], []).append(f)
    for v in per.values():
        v.sort()
    keys = sorted(per, key=lambda d: int(d.rsplit("_", 1)[1]))
    files = [per[k][i] for i in range(max(len(v) for v in per.values()))
             for k in keys if i < len(per[k])]
    seqs, t0, ndoc = [], time.time(), 0
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset")
        tbl = pq.read_table(p, columns=["text", "language", "language_type"])
        rows = tbl.to_pylist()
        texts = [r["text"] for r in rows if keep(r) and r["text"]]
        ndoc += len(texts)
        done = False
        for i in range(0, len(texts), 2048):
            if pack(tokenizer, texts[i:i + 2048], seqs, target, None):
                done = True
                break
        print(f"  [{tag}] {f}: {len(texts)}/{len(rows)} docs kept -> {len(seqs)}/{target} seqs "
              f"({time.time() - t0:.0f}s)", flush=True)
        Path(p).unlink(missing_ok=True)        # 430MB each; do not hoard the whole corpus
        if done:
            break
    assert len(seqs) == target, f"only reached {len(seqs)} of {target} sequences"
    arr = np.stack(seqs)
    arr.tofile(OUT / f"{tag}.bin")
    print(f"[{tag}] {arr.shape} tokens={arr.size} from {ndoc} documents")
    return {"seqs": int(arr.shape[0]), "tokens": int(arr.size), "docs": ndoc, "dirs": dirs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=95464, help="the note's train sequence count")
    ap.add_argument("--val-seqs", type=int, default=4096)
    ap.add_argument("--check", action="store_true", help="tokenizer sanity only, no download")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tk = build_tokenizer(OUT / "tok4096.json")

    s = "ACETYLCHOLINE"
    print(f"  {s} -> {tk.encode(s).tokens}   (the note's ' IN ' -> ' E ' example)")
    if args.check:
        return

    meta = {"vocab": VOCAB, "ctx": CTX, "delim": DELIM,
            "train": build_split(tk, [f"common_corpus_{i}" for i in range(1, 10)], args.seqs, "train"),
            "val": build_split(tk, ["common_corpus_10"], args.val_seqs, "val")}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

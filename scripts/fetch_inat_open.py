#!/usr/bin/env python
"""Enlarge a taxon manifest with CC-licensed iNaturalist open-data photos.

    python scripts/fetch_inat_open.py data/manifests/herps --out data/manifests/herps_plus --cap 600

Phases, each resumable from its files under data/datasets/inat_open/<taxon>/:
  resolve   label -> iNaturalist taxon id (api.inaturalist.org/v1/taxa, exact name match)
  list      research-grade observations with CC0 / CC-BY / CC-BY-NC photos, up to --cap photos
            per species, at most two per observation (api.inaturalist.org/v1/observations)
  download  medium (500 px) files from the inaturalist-open-data S3 bucket, 16 at a time
  dedupe    drop any download within Hamming 6 (dHash) of a validation image, and exact repeats
  manifest  <out>/train.tsv = the source train rows + new rows; val.tsv and labels.txt copied

New rows get image_id 1_000_000_000 + photo_id and a path relative to the iNat2021 root
("../inat_open/<taxon>/<label>/<photo_id>.jpg") so distill.py needs no new flag.
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT, load_finetuned_labels  # noqa: E402
from teacher_labels import read_manifest  # noqa: E402

API = "https://api.inaturalist.org/v1"
LICENSES = {"cc0", "cc-by", "cc-by-nc"}
UA = {"User-Agent": "species-classifiers (github.com/mlgraham/species-classifiers)"}
_last_call = [0.0]


def api(path, **params):
    """One request per second, retries on 429 and 5xx."""
    for attempt in range(6):
        wait = 1.0 - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        r = requests.get(f"{API}/{path}", params=params, headers=UA, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up on {path} {params}")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve(labels, path):
    done = json.load(open(path)) if os.path.exists(path) else {}
    for i, label in enumerate(labels):
        if str(i) in done:
            continue
        name = label.split(" (")[0]
        hits = api("taxa", q=name, rank="species", per_page=5).get("results", [])
        exact = [t for t in hits if t["name"].lower() == name.lower()]
        done[str(i)] = exact[0]["id"] if exact else None
        if not exact:
            log(f"unresolved: {label}")
        if i % 50 == 0:
            json.dump(done, open(path, "w"))
    json.dump(done, open(path, "w"))
    return done


def list_photos(taxa, cap, path):
    seen_taxa = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                seen_taxa.add(json.loads(line)["label"])
    with open(path, "a") as out:
        for label, taxon_id in taxa.items():
            label = int(label)
            if label in seen_taxa or taxon_id is None:
                continue
            got, page = 0, 1
            while got < cap and page * 200 <= 10000:
                res = api("observations", taxon_id=taxon_id, quality_grade="research", photos="true",
                          photo_license=",".join(LICENSES), per_page=200, page=page)
                results = res.get("results", [])
                for o in results:
                    if o.get("taxon", {}).get("id") != taxon_id:
                        continue  # the query includes descendants; keep the species itself only
                    n = 0
                    for p in o.get("photos", []):
                        if p.get("license_code") not in LICENSES or n >= 2 or got >= cap:
                            continue
                        url = p["url"].replace("/square.", "/medium.")
                        out.write(json.dumps({"label": label, "photo_id": p["id"], "url": url,
                                              "license": p["license_code"], "observation": o["id"]}) + "\n")
                        got += 1
                        n += 1
                if len(results) < 200:
                    break
                page += 1
            out.flush()
            log(f"label {label:3d} taxon {taxon_id}: {got} photos")


def download(rows, root):
    sess = requests.Session()
    sess.headers.update(UA)

    def one(r):
        dest = os.path.join(root, str(r["label"]), f"{r['photo_id']}.jpg")
        if os.path.exists(dest):
            return True
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        for attempt in range(3):
            try:
                resp = sess.get(r["url"], timeout=60)
                if resp.status_code == 200 and resp.content:
                    tmp = dest + ".part"
                    with open(tmp, "wb") as f:
                        f.write(resp.content)
                    Image.open(tmp).verify()
                    os.replace(tmp, dest)
                    return True
                if resp.status_code == 404:
                    return False
            except Exception:
                pass
            time.sleep(2 * (attempt + 1))
        return False

    ok = 0
    with ThreadPoolExecutor(16) as ex:
        for i, good in enumerate(ex.map(one, rows), 1):
            ok += bool(good)
            if i % 5000 == 0:
                log(f"downloaded {i}/{len(rows)} ({ok} ok)")
    log(f"download done: {ok}/{len(rows)} files")


def dhash(path):
    try:
        img = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def popcount64(x):
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="source manifest dir, e.g. data/manifests/herps")
    ap.add_argument("--out", required=True, help="new manifest dir, e.g. data/manifests/herps_plus")
    ap.add_argument("--cap", type=int, default=600, help="new photos per species at most")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "datasets", "inat2021"))
    ap.add_argument("--limit-taxa", type=int, help="smoke test: only the first N labels")
    args = ap.parse_args()

    taxon = os.path.basename(os.path.normpath(args.manifest))
    work = os.path.join(ROOT, "data", "datasets", "inat_open", taxon)
    os.makedirs(work, exist_ok=True)
    labels = load_finetuned_labels(os.path.join(args.manifest, "labels.txt"))
    if args.limit_taxa:
        labels = labels[: args.limit_taxa]

    log(f"resolve {len(labels)} labels")
    taxa = resolve(labels, os.path.join(work, "taxa.json"))
    log(f"resolved {sum(v is not None for v in taxa.values())}/{len(taxa)}")

    photos_path = os.path.join(work, "photos.jsonl")
    log(f"list photos, cap {args.cap} per species")
    list_photos(taxa, args.cap, photos_path)
    rows = [json.loads(l) for l in open(photos_path)]
    log(f"{len(rows)} photos listed")

    log("download")
    download(rows, work)

    log("dedupe against val")
    val_rows = read_manifest(os.path.join(args.manifest, "val.tsv"))
    val_h = np.array([h for h in (dhash(os.path.join(args.root, rel)) for _, rel, _ in val_rows) if h is not None], dtype=np.uint64)
    kept, dropped_val, dropped_dup, missing, seen = [], 0, 0, 0, set()
    for start in range(0, len(rows), 2000):
        chunk = rows[start:start + 2000]
        hs = [dhash(os.path.join(work, str(r["label"]), f"{r['photo_id']}.jpg")) for r in chunk]
        for r, h in zip(chunk, hs):
            if h is None:
                missing += 1
                continue
            if h in seen:
                dropped_dup += 1
                continue
            d = popcount64(np.bitwise_xor(val_h, np.uint64(h)))
            if d.size and int(d.min()) <= 6:
                dropped_val += 1
                continue
            seen.add(h)
            kept.append(r)
    log(f"kept {len(kept)}, dropped {dropped_val} near val, {dropped_dup} repeats, {missing} unreadable or missing")

    os.makedirs(args.out, exist_ok=True)
    train_rows = read_manifest(os.path.join(args.manifest, "train.tsv"))
    with open(os.path.join(args.out, "train.tsv"), "w", encoding="utf-8") as f:
        for image_id, rel, label in train_rows:
            f.write(f"{image_id}\t{rel}\t{label}\n")
        for r in kept:
            f.write(f"{1_000_000_000 + r['photo_id']}\t../inat_open/{taxon}/{r['label']}/{r['photo_id']}.jpg\t{r['label']}\n")
    for name in ("val.tsv", "labels.txt"):
        shutil.copy(os.path.join(args.manifest, name), os.path.join(args.out, name))
    per = {}
    for r in kept:
        per[r["label"]] = per.get(r["label"], 0) + 1
    counts = sorted(per.values())
    json.dump({"source_train": len(train_rows), "new": len(kept), "species_with_new": len(per),
               "new_per_species_min": counts[0] if counts else 0, "new_per_species_median": counts[len(counts) // 2] if counts else 0,
               "dropped_near_val": dropped_val, "dropped_repeats": dropped_dup, "cap": args.cap},
              open(os.path.join(args.out, "fetch_summary.json"), "w"), indent=1)
    log(f"wrote {args.out}/train.tsv: {len(train_rows)} + {len(kept)} = {len(train_rows) + len(kept)} rows")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build a training manifest for one iNat2021 supercategory.

    python scripts/inat_subset.py Mammals
    python scripts/inat_subset.py Arachnids --root data/datasets/inat2021
    python scripts/inat_subset.py Reptiles Amphibians --train-split train --out data/manifests/herps
    python scripts/inat_subset.py Insects --include order:Hymenoptera order:Mantodea family:Asilidae \
        family:Syrphidae --train-split train --out data/manifests/hive

--include keeps only the species whose order, family or genus matches one of
the given rank:name filters (any match keeps the species).

Reads train_mini.json and val.json from the dataset root and writes, under
data/manifests/<slug>/:

    labels.txt   one class per line, "Latin name (Common name)", index = class id
    train.tsv    image_id <tab> path <tab> label
    val.tsv      same, from the official val split (10 images per species)
    classes.json the category records, for taxonomy roll-ups later

Paths are relative to the dataset root, exactly as the JSON gives them, so
the manifest works before the image tarballs are extracted.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT  # noqa: E402

SUPERCATEGORIES = [
    "Mammals", "Arachnids", "Reptiles", "Amphibians", "Fungi", "Birds", "Insects",
    "Plants", "Mollusks", "Ray-finned Fishes", "Animalia",
]


def load(root, split):
    with open(os.path.join(root, f"{split}.json"), encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("supercategory", nargs="+", choices=SUPERCATEGORIES, help="one or more; several are merged into one label space")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "datasets", "inat2021"))
    ap.add_argument("--out", help="manifest dir (default data/manifests/<slug>)")
    ap.add_argument("--train-split", default="train_mini", choices=["train_mini", "train"],
                    help="which annotation file feeds train.tsv: train_mini (50 per species) or the full train")
    ap.add_argument("--include", nargs="+", metavar="RANK:NAME",
                    help="keep only species matching any filter, e.g. order:Hymenoptera family:Syrphidae genus:Vespa")
    args = ap.parse_args()

    filters = []
    for item in args.include or []:
        rank, _, name = item.partition(":")
        if rank not in ("order", "family", "genus") or not name:
            sys.exit(f"--include wants order:, family: or genus: filters, got {item!r}")
        filters.append((rank, name))

    def keep(c):
        return not filters or any((c["name"].split()[0] if rank == "genus" else c[rank]) == name for rank, name in filters)

    wanted = set(args.supercategory)
    slug = "_".join(c.lower().replace(" ", "_").replace("-", "_") for c in args.supercategory)
    out = args.out or os.path.join(ROOT, "data", "manifests", slug)
    os.makedirs(out, exist_ok=True)

    val = load(args.root, "val")
    cats = sorted((c for c in val["categories"] if c["supercategory"] in wanted and keep(c)), key=lambda c: c["id"])
    index = {c["id"]: i for i, c in enumerate(cats)}
    with open(os.path.join(out, "labels.txt"), "w", encoding="utf-8") as f:
        for c in cats:
            common = c.get("common_name") or ""
            f.write(f"{c['name']} ({common})\n" if common else f"{c['name']}\n")
    with open(os.path.join(out, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(cats, f, indent=1)

    counts = {}
    for split, data in (("val", val), ("train", load(args.root, args.train_split))):
        images = {im["id"]: im["file_name"] for im in data["images"]}
        rows = []
        for a in data["annotations"]:
            label = index.get(a["category_id"])
            if label is not None:
                rows.append((a["image_id"], images[a["image_id"]], label))
        rows.sort()
        with open(os.path.join(out, f"{split}.tsv"), "w", encoding="utf-8") as f:
            for image_id, path, label in rows:
                f.write(f"{image_id}\t{path}\t{label}\n")
        counts[split] = len(rows)

    print(f"{' + '.join(args.supercategory)}{' (' + ' '.join(args.include) + ')' if filters else ''}: {len(cats)} classes, train {counts['train']} images ({args.train_split}), val {counts['val']} images -> {out}")


if __name__ == "__main__":
    main()

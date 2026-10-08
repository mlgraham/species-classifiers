#!/usr/bin/env python
"""Measure a stock AIY model on iNat2021 val, for the species the two label sets share.

    python scripts/eval_stock_on_inat2021.py insects_V1
    python scripts/eval_stock_on_inat2021.py birds_V1 --model models/tflite/birds_V1_int8.tflite

The stock models were trained by Google on iNat2017 with their own species
lists (1021 insects, 2101 plants, 964 birds). iNat2021's lists differ, so
this matches Latin names between the two, evaluates the stock model on the
val images of matched species only, and scores a prediction as correct when
the predicted stock label's Latin name equals the true species. The stock
model still chooses among all its classes, so this is the honest number for
"how good is the stock model on today's photos", not a like-for-like race
against a student with fewer classes.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_tflite import load_image  # noqa: E402
from labels import ROOT, load_labels  # noqa: E402

SUPER = {"insects_V1": "Insects", "plants_V1": "Plants", "birds_V1": "Birds"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=list(SUPER))
    ap.add_argument("--model", help="TFLite file (default models/tflite/<name>_int8.tflite)")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "datasets", "inat2021"))
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    import tensorflow as tf

    stock = [l.split(" (")[0] for l in load_labels(args.name)]
    stock_index = {n: i for i, n in enumerate(stock)}
    with open(os.path.join(args.root, "val.json"), encoding="utf-8") as f:
        val = json.load(f)
    cats = {c["id"]: c for c in val["categories"] if c["supercategory"] == SUPER[args.name]}
    matched = {cid: c["name"] for cid, c in cats.items() if c["name"] in stock_index}
    images = {im["id"]: im["file_name"] for im in val["images"]}
    rows = [(images[a["image_id"]], matched[a["category_id"]]) for a in val["annotations"] if a["category_id"] in matched]
    rows.sort()
    if args.limit:
        rows = rows[: args.limit]
    print(f"{args.name}: {len(stock)} stock classes, {len(cats)} iNat2021 {SUPER[args.name].lower()} species, "
          f"{len(matched)} share a Latin name, {len(rows)} val images")

    model_path = args.model or os.path.join(ROOT, "models", "tflite", f"{args.name}_int8.tflite")
    interp = tf.lite.Interpreter(model_path=model_path, num_threads=8)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    size = int(inp["shape"][1])
    top1 = top5 = 0
    t0 = time.time()
    for rel, name in rows:
        x = load_image(os.path.join(args.root, rel), size, False, 0.875).astype(inp["dtype"])[None]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        scores = interp.get_tensor(out["index"])[0].astype(np.float32)
        order = np.argsort(-scores)
        want = stock_index[name]
        top1 += int(order[0] == want)
        top5 += int(want in order[:5])
    n = len(rows)
    print(f"{os.path.basename(model_path)} on matched iNat2021 val ({n} images, {len(matched)} species): "
          f"top-1 {top1 / n:.4f}  top-5 {top5 / n:.4f}  {(time.time() - t0) * 1000 / n:.1f} ms/image")


if __name__ == "__main__":
    main()

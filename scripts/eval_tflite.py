#!/usr/bin/env python
"""Measure a TFLite file's top-1 on a manifest's val split, the way a device would run it.

    python scripts/eval_tflite.py models/tflite/mammals_full_w100_int8.tflite data/manifests/mammals_full
    python scripts/eval_tflite.py models/tflite/mammals_full_w100_int8.tflite data/manifests/mammals_full --grayscale

Feeds uint8 RGB (or float32 for the float export), center-cropped and
resized to the model's input size by preprocess.center_crop, exactly as
classify.py and webcam.py do. Prints top-1 and top-5. This is the number to
quote for the shipped file; the PyTorch checkpoint's number is the recipe's.
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT  # noqa: E402
from preprocess import CROP_FRAC, center_crop  # noqa: E402
from teacher_labels import read_manifest  # noqa: E402


def load_image(path, size, grayscale, crop_frac=CROP_FRAC):
    """uint8 RGB array of preprocess.center_crop: the 87.5% center crop by default, the same one
    distill.py validates with. --crop-frac 1.0 keeps the whole image, squared."""
    img = Image.open(path).convert("RGB")
    if grayscale:
        img = img.convert("L").convert("RGB")
    return np.asarray(center_crop(img, size, crop_frac))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("manifest")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "datasets", "inat2021"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--crop-frac", type=float, default=CROP_FRAC, help="center-crop fraction; 1.0 = whole image squared")
    args = ap.parse_args()

    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=args.model, num_threads=8)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    size = int(inp["shape"][1])
    rows = read_manifest(os.path.join(args.manifest, f"{args.split}.tsv"))
    if args.limit:
        rows = rows[: args.limit]

    top1 = top5 = 0
    t0 = time.time()
    for _, rel, label in rows:
        x = load_image(os.path.join(args.root, rel), size, args.grayscale, args.crop_frac)
        x = x.astype(inp["dtype"])[None]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        scores = interp.get_tensor(out["index"])[0].astype(np.float32)  # uint8 would wrap under negation
        order = np.argsort(-scores)
        top1 += int(order[0] == label)
        top5 += int(label in order[:5])
    n = len(rows)
    ms = (time.time() - t0) * 1000 / n
    mode = ("grayscale" if args.grayscale else "color") + f", crop {args.crop_frac}"
    print(f"{os.path.basename(args.model)} on {args.manifest} {args.split} ({mode}, {n} images): "
          f"top-1 {top1 / n:.4f}  top-5 {top5 / n:.4f}  {ms:.1f} ms/image on CPU")


if __name__ == "__main__":
    main()

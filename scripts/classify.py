#!/usr/bin/env python
"""Classify one or more images from the command line.

    python scripts/classify.py data/samples/*.jpg
    python scripts/classify.py --model models/keras/plants_V1.keras --name plants_V1 leaf.jpg
    python scripts/classify.py --model models/keras/my_finetune.keras --labels models/keras/my_finetune_labels.txt photo.jpg

Run scripts/rebuild_keras.py first to create models/keras/<name>.keras.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT, load_finetuned_labels, load_labels  # noqa: E402


def load_image(path, size=224):
    img = Image.open(path).convert("RGB")
    # centre crop to square so the subject is not squashed
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--name", default="insects_V1", choices=["insects_V1", "plants_V1", "birds_V1"])
    ap.add_argument("--model", help="path to a .keras model (default models/keras/<name>.keras)")
    ap.add_argument("--labels", help="one-label-per-line file from finetune.py (default: stock labelmap)")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    import tensorflow as tf

    model_path = args.model or os.path.join(ROOT, "models", "keras", f"{args.name}.keras")
    model = tf.keras.models.load_model(model_path)
    labels = load_finetuned_labels(args.labels) if args.labels else load_labels(args.name)

    batch = np.stack([load_image(p) for p in args.images])
    probs = model.predict(batch, verbose=0)
    for path, p in zip(args.images, probs):
        print(path)
        for i in np.argsort(-p)[: args.top]:
            print(f"  {p[i]:6.3f}  {labels[i]}")


if __name__ == "__main__":
    main()

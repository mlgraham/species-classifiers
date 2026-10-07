#!/usr/bin/env python
"""Rebuild Google's AIY iNaturalist classifier as a trainable Keras model.

The TF Hub modules (models/tfhub/*_V1) are TF1 "hub module" graphs whose
weights are baked in as Const nodes, so they can be run but not trained.
This script pulls those constants out and loads them into a stock
tf.keras MobileNetV2 (alpha=1.0, 224x224) plus a softmax head, giving a
model you can fine-tune, export to TFLite, or quantize for an Edge TPU.

Output: models/keras/<name>.keras

The rebuilt model takes uint8/float RGB images in the 0..255 range
(a Rescaling layer maps them to -1..1 internally), and its output is the
same 1022-way probability vector the Hub module produces.

Usage:
    python scripts/rebuild_keras.py insects_V1
    python scripts/rebuild_keras.py plants_V1 --no-verify
"""
import argparse
import csv
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_constants(saved_model_path):
    sm = saved_model_pb2.SavedModel()
    with open(saved_model_path, "rb") as f:
        sm.ParseFromString(f.read())
    gd = sm.meta_graphs[0].graph_def
    consts = {}
    order = []
    for n in gd.node:
        if n.op != "Const":
            continue
        t = n.attr["value"].tensor
        if not t.tensor_shape.dim:
            continue  # scalars (quant ranges, epsilons) are not weights
        if "_quant" in n.name or "BatchNorm_Fold" in n.name:
            continue  # fake-quant bookkeeping, not weights
        if not (n.name.startswith("MobilenetV2/") or n.name.startswith("Logits/")):
            continue
        consts[n.name] = tf.make_ndarray(t)
        order.append(n.name)
    return consts, order


def build_keras(num_classes):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), alpha=1.0, include_top=False, weights=None
    )
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    return tf.keras.Model(inputs, outputs, name="aiy_inat_mobilenet_v2"), backbone


def transplant(consts, order, backbone, head):
    """Copy slim-named constants into Keras weights.

    Both TF-slim's mobilenet_v2 and tf.keras's MobileNetV2 emit weights in
    the same block order with the same (gamma, beta, mean, variance) batch
    norm layout, so a positional copy with shape checks is sufficient.
    """
    slim_backbone = [n for n in order if n.startswith("MobilenetV2/")]
    keras_weights = backbone.weights
    if len(slim_backbone) != len(keras_weights):
        sys.exit(f"tensor count mismatch: slim={len(slim_backbone)} keras={len(keras_weights)}")
    values = []
    for name, w in zip(slim_backbone, keras_weights):
        v = consts[name]
        if tuple(v.shape) != tuple(w.shape):
            sys.exit(f"shape mismatch at {name}: slim={v.shape} keras={w.shape}")
        values.append(v)
    backbone.set_weights(values)
    head.set_weights([consts["Logits/weights"], consts["Logits/biases"]])


def verify(model, module_dir, sample_dir):
    import tensorflow_hub as hub
    from PIL import Image

    hub_fn = hub.load(module_dir).signatures["default"]
    samples = sorted(
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not samples:
        print("no sample images found, skipping verification")
        return
    worst = 0.0
    for p in samples:
        img = np.asarray(Image.open(p).convert("RGB").resize((224, 224)), dtype=np.float32)
        ref = hub_fn(images=tf.constant(img[None] / 255.0))["default"].numpy()[0]
        out = model.predict(img[None], verbose=0)[0]
        diff = float(np.abs(ref - out).max())
        worst = max(worst, diff)
        print(f"  {os.path.basename(p)}: argmax hub={ref.argmax()} keras={out.argmax()} max|diff|={diff:.4f}")
    print(f"verification: max abs difference vs Hub module = {worst:.4f}")
    if worst > 0.05:
        print("WARNING: outputs differ more than expected; check the weight mapping")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=["insects_V1", "plants_V1", "birds_V1"])
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    module_dir = os.path.join(ROOT, "models", "tfhub", args.name)
    consts, order = load_constants(os.path.join(module_dir, "saved_model.pb"))
    num_classes = consts["Logits/biases"].shape[0]
    with open(os.path.join(module_dir, "labelmap.csv")) as f:
        n_labels = sum(1 for _ in csv.DictReader(f))
    print(f"{args.name}: {len(order)} weight tensors, {num_classes} classes, {n_labels} labels")

    model, backbone = build_keras(num_classes)
    transplant(consts, order, backbone, model.get_layer("logits"))

    out_dir = os.path.join(ROOT, "models", "keras")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}.keras")
    model.save(out_path)
    print(f"saved {out_path} ({model.count_params():,} params)")

    if not args.no_verify:
        verify(model, module_dir, os.path.join(ROOT, "data", "samples"))


if __name__ == "__main__":
    main()

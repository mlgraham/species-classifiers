#!/usr/bin/env python
"""Transplant a distill.py student into Keras so export_tflite.py can ship it.

    python scripts/torch_to_keras.py models/torch/mammals.pt
    python scripts/export_tflite.py models/keras/mammals.keras --rep-dir data/datasets/inat2021/train_mini

Writes models/keras/<stem>.keras and <stem>_labels.txt, then checks the
Keras model against the PyTorch one on the sample images and reports the
largest probability difference. timm's mobilenetv2 (with pad_type="same")
and tf.keras MobileNetV2 lay their weights out in the same block order, so
the copy is positional with shape checks and tensor transposition.

The Keras model takes 0..255 RGB (Rescaling inside), same as everything else.
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT  # noqa: E402


def torch_weights_in_order(state):
    """Yield (name, array) for conv/bn/classifier params in timm's forward order, Keras-shaped."""
    out = []
    keys = list(state.keys())
    for k in keys:
        v = state[k]
        if k.endswith("num_batches_tracked"):
            continue
        a = v.detach().cpu().numpy()
        if a.ndim == 4:  # conv: torch [out, in/groups, kh, kw]
            if a.shape[1] == 1 and a.shape[0] > 1 and "conv_dw" in k:
                a = a.transpose(2, 3, 0, 1)  # depthwise -> [kh, kw, C, 1]
            else:
                a = a.transpose(2, 3, 1, 0)  # -> [kh, kw, in, out]
        elif a.ndim == 2:  # classifier [C, 1280] -> [1280, C]
            a = a.T
        out.append((k, a))
    return out


BN_VAR_FLOOR = 1e-3


def floor_dead_bn_channels(backbone):
    """Optional. Dead channels (running variance ~0) fold into weights scaled by gamma/sqrt(eps); this floors
    them before export. Kept behind --bn-floor because it was measured NOT to help: on mammals_full_w100,
    int8 top-1 on 400 val images was 0.5450 without the floor and 0.5275 with it. (An apparent 0.0% int8
    result that motivated it was an evaluator bug: negating a uint8 score vector.)"""
    count = 0
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            gamma, beta, mean, var = [w.numpy() for w in layer.weights]
            dead = var < BN_VAR_FLOOR
            if dead.any():
                layer.set_weights([gamma, beta, mean, np.maximum(var, BN_VAR_FLOOR)])
                count += int(dead.sum())
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--bn-floor", action="store_true", help="floor near-zero BN running variances before export (measured: not needed, costs ~1.5 points int8)")
    ap.add_argument("--out-dir", help="where to write the .keras (default models/keras)")
    args = ap.parse_args()

    import timm

    ck = torch.load(args.checkpoint, map_location="cpu")
    cfg, labels = ck["config"], ck["labels"]
    size, width = cfg["size"], cfg["width"]
    alpha = {"035": 0.35, "050": 0.5, "075": 0.75, "100": 1.0}[width]

    student = timm.create_model(f"mobilenetv2_{width}", pretrained=False, num_classes=len(labels), pad_type="same")
    for module in student.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = ck.get("bn_eps", 1e-5)
    student.load_state_dict(ck["state_dict"])
    student.eval()

    backbone = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=alpha, include_top=False, weights=None)
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.epsilon = ck.get("bn_eps", 1e-5)  # timm default 1e-5; iNat-initialised students keep Keras's 1e-3
    inputs = tf.keras.Input(shape=(size, size, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax", name="logits")(x)
    model = tf.keras.Model(inputs, outputs, name=f"species_student_{cfg['out']}")

    src = torch_weights_in_order(student.state_dict())
    dst = backbone.weights + model.get_layer("logits").weights
    if len(src) != len(dst):
        sys.exit(f"tensor count mismatch: torch={len(src)} keras={len(dst)}")
    values = []
    for (name, a), w in zip(src, dst):
        if tuple(a.shape) != tuple(w.shape):
            sys.exit(f"shape mismatch at {name} -> {w.path if hasattr(w, 'path') else w.name}: {a.shape} vs {w.shape}")
        values.append(a)
    backbone.set_weights(values[: len(backbone.weights)])
    model.get_layer("logits").set_weights(values[len(backbone.weights):])
    if args.bn_floor:
        floored = floor_dead_bn_channels(backbone)
        print(f"batch-norm channels with running variance under {BN_VAR_FLOOR} floored: {floored}")

    out_dir = args.out_dir or os.path.join(ROOT, "models", "keras")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    model.save(os.path.join(out_dir, f"{stem}.keras"))
    with open(os.path.join(out_dir, f"{stem}_labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")
    print(f"saved {os.path.join(out_dir, stem + '.keras')} and {stem}_labels.txt ({model.count_params():,} params)")

    if args.no_verify:
        return
    from PIL import Image
    sample_dir = os.path.join(ROOT, "data", "samples")
    worst = 0.0
    for fn in sorted(os.listdir(sample_dir)):
        if not fn.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img = Image.open(os.path.join(sample_dir, fn)).convert("RGB").resize((size, size))
        a = np.asarray(img, dtype=np.float32)
        k = model.predict(a[None], verbose=0)[0]
        with torch.no_grad():
            t = torch.softmax(student(torch.from_numpy((a / 127.5 - 1.0).transpose(2, 0, 1)[None].copy())), 1)[0].numpy()
        diff = float(np.abs(k - t).max())
        worst = max(worst, diff)
        print(f"  {fn}: argmax torch={t.argmax()} keras={k.argmax()} max|diff|={diff:.5f}")
    print(f"verification: max abs probability difference = {worst:.5f}")
    if worst > 1e-3:
        print("WARNING: transplant differs from the PyTorch model; do not ship this file")


if __name__ == "__main__":
    main()

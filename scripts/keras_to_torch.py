#!/usr/bin/env python
"""Turn a stock Keras model's backbone into a PyTorch init for distill.py.

    python scripts/keras_to_torch.py insects_V1
    python scripts/distill.py data/manifests/mammals --out mammals_inat --width 100 --init models/torch/init_insects_V1.pt

The stock AIY models are MobileNetV2 1.0 trained on iNaturalist, so their
backbone is a better starting point for a species student than ImageNet.
This is the inverse of torch_to_keras.py: positional copy from the Keras
backbone into timm's mobilenetv2_100 (pad_type="same", matching the TF
padding the Keras model was trained with), with BN epsilon kept at the
Keras value so the two produce the same features.

Writes models/torch/init_<name>.pt holding the backbone state_dict and
verifies the pooled 1280-d features against the Keras model on the
sample images.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT  # noqa: E402

BN_EPS_KERAS = 1e-3


def keras_backbone_arrays(keras_model):
    backbone = keras_model.get_layer("mobilenetv2_1.00_224")
    return [w.numpy() for w in backbone.weights]


def to_torch_shape(array, torch_key, torch_shape):
    if array.ndim == 4:
        if "conv_dw" in torch_key:
            out = array.transpose(2, 3, 0, 1)  # [kh,kw,C,1] -> [C,1,kh,kw]
        else:
            out = array.transpose(3, 2, 0, 1)  # [kh,kw,in,out] -> [out,in,kh,kw]
    else:
        out = array
    if tuple(out.shape) != tuple(torch_shape):
        raise SystemExit(f"shape mismatch at {torch_key}: keras {array.shape} -> {out.shape} vs torch {tuple(torch_shape)}")
    return torch.from_numpy(np.ascontiguousarray(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=["insects_V1", "plants_V1", "birds_V1"])
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    import tensorflow as tf
    import timm

    keras_path = os.path.join(ROOT, "models", "keras", f"{args.name}.keras")
    if not os.path.exists(keras_path):
        sys.exit(f"{keras_path} missing: run scripts/rebuild_keras.py {args.name} first")
    keras_model = tf.keras.models.load_model(keras_path)
    arrays = keras_backbone_arrays(keras_model)

    student = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=0, pad_type="same")
    for module in student.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = BN_EPS_KERAS
    state = student.state_dict()
    keys = [k for k in state if not k.endswith("num_batches_tracked")]
    if len(keys) != len(arrays):
        sys.exit(f"tensor count mismatch: keras {len(arrays)} vs torch {len(keys)}")
    new_state = {k: to_torch_shape(a, k, state[k].shape) for k, a in zip(keys, arrays)}
    student.load_state_dict(new_state, strict=False)
    student.eval()

    out_dir = os.path.join(ROOT, "models", "torch")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"init_{args.name}.pt")
    torch.save({"backbone": student.state_dict(), "bn_eps": BN_EPS_KERAS, "source": args.name, "width": "100"}, out_path)
    print(f"wrote {out_path} ({sum(v.numel() for v in new_state.values()):,} values)")

    if args.no_verify:
        return
    from PIL import Image
    feat_model = tf.keras.Model(keras_model.input, keras_model.get_layer("avg_pool").output)
    worst = 0.0
    for fn in sorted(os.listdir(os.path.join(ROOT, "data", "samples"))):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = np.asarray(Image.open(os.path.join(ROOT, "data", "samples", fn)).convert("RGB").resize((224, 224)), dtype=np.float32)
        kf = feat_model.predict(img[None], verbose=0)[0]
        with torch.no_grad():
            tf_ = student(torch.from_numpy((img / 127.5 - 1.0).transpose(2, 0, 1)[None].copy()))[0].numpy()
        diff = float(np.abs(kf - tf_).max())
        worst = max(worst, diff)
        print(f"  {fn}: pooled features max|diff| {diff:.2e} (scale {np.abs(kf).mean():.3f})")
    print(f"verification: max abs feature difference = {worst:.2e}")
    if worst > 1e-3:
        print("WARNING: transplant differs from the Keras model")


if __name__ == "__main__":
    main()

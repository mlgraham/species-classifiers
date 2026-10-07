#!/usr/bin/env python
"""Export a Keras model to TFLite, including the int8 form an Edge TPU needs.

    python scripts/export_tflite.py models/keras/insects_V1.keras
    python scripts/export_tflite.py models/keras/backyard.keras --rep-dir data/datasets/backyard

Writes models/tflite/<stem>_float32.tflite and <stem>_int8.tflite.

The int8 file is full-integer quantized with uint8 input and output, which
is what the Edge TPU compiler and the Coral Dev Board Micro expect. The
representative images steer the activation ranges, so point --rep-dir at
photos that look like what the device will see (a few hundred is plenty).
Stock models were trained quantization-aware, so int8 accuracy holds up.

Then, on Linux x86-64 (Docker works on an Intel Mac):

    edgetpu_compiler models/tflite/<stem>_int8.tflite

See README.md for the Docker invocation and the Dev Board Micro steps.
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT  # noqa: E402


def representative_images(rep_dir, size, limit):
    paths = sorted(
        p for ext in ("jpg", "jpeg", "png", "JPG")
        for p in glob.glob(os.path.join(rep_dir, "**", f"*.{ext}"), recursive=True)
    )[:limit]
    if not paths:
        sys.exit(f"no images found under {rep_dir}")
    print(f"using {len(paths)} representative images from {rep_dir}")

    def gen():
        for p in paths:
            img = Image.open(p).convert("RGB").resize((size, size))
            yield [np.asarray(img, dtype=np.float32)[None]]

    return gen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--rep-dir", default=os.path.join(ROOT, "data", "samples"))
    ap.add_argument("--rep-limit", type=int, default=300)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "models", "tflite"))
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)
    size = model.input_shape[1]
    stem = os.path.splitext(os.path.basename(args.model))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    # Keras 3 models loaded from .keras trip up from_keras_model in TF 2.16
    # ("ReadVariableOp missing attribute 'value'"), so go through a SavedModel.
    export_dir = tempfile.mkdtemp(prefix="tflite_export_")
    model.export(export_dir)

    conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    f32 = os.path.join(args.out_dir, f"{stem}_float32.tflite")
    with open(f32, "wb") as f:
        f.write(conv.convert())
    print(f"wrote {f32} ({os.path.getsize(f32) / 1e6:.1f} MB)")

    conv = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_images(args.rep_dir, size, args.rep_limit)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    i8 = os.path.join(args.out_dir, f"{stem}_int8.tflite")
    with open(i8, "wb") as f:
        f.write(conv.convert())
    print(f"wrote {i8} ({os.path.getsize(i8) / 1e6:.1f} MB)")

    # quick sanity check: run both on the first representative image
    interp = tf.lite.Interpreter(model_path=i8)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    sample = next(iter(representative_images(args.rep_dir, size, 1)()))[0]
    interp.set_tensor(inp["index"], sample.astype(np.uint8))
    interp.invoke()
    q = interp.get_tensor(out["index"])[0]
    ref = model.predict(sample, verbose=0)[0]
    print(f"int8 argmax={q.argmax()} float argmax={ref.argmax()} (input {inp['dtype'].__name__} {inp['shape']})")
    shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

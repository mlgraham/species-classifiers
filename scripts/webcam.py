#!/usr/bin/env python
"""Live classification from the laptop webcam.

    python scripts/webcam.py                         # insects, stock model
    python scripts/webcam.py --name birds_V1
    python scripts/webcam.py --model models/keras/my_finetune.keras --labels models/keras/my_finetune_labels.txt

Keys:  q quit   s save the current frame to data/captures/

macOS will ask for camera permission for your terminal app the first time.
Hold a printed photo or a phone screen in front of the camera to try it.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT, load_finetuned_labels, load_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="insects_V1", choices=["insects_V1", "plants_V1", "birds_V1"])
    ap.add_argument("--model")
    ap.add_argument("--labels")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.15, help="hide predictions below this probability")
    args = ap.parse_args()

    import tensorflow as tf

    model_path = args.model or os.path.join(ROOT, "models", "keras", f"{args.name}.keras")
    model = tf.keras.models.load_model(model_path)
    labels = load_finetuned_labels(args.labels) if args.labels else load_labels(args.name)
    size = model.input_shape[1]

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("could not open camera; check System Settings > Privacy & Security > Camera")
    os.makedirs(os.path.join(ROOT, "data", "captures"), exist_ok=True)

    smoothed = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = frame[y0 : y0 + s, x0 : x0 + s]
        rgb = cv2.cvtColor(cv2.resize(crop, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32)

        t0 = time.time()
        probs = model.predict(rgb[None], verbose=0)[0]
        ms = (time.time() - t0) * 1000
        smoothed = probs if smoothed is None else 0.7 * smoothed + 0.3 * probs

        cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (80, 80, 80), 1)
        y = 30
        for i in np.argsort(-smoothed)[:3]:
            if smoothed[i] < args.threshold:
                continue
            cv2.putText(frame, f"{smoothed[i]:.2f} {labels[i]}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 28
        cv2.putText(frame, f"{ms:.0f} ms", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("species-classifiers", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = os.path.join(ROOT, "data", "captures", f"{int(time.time())}.jpg")
            cv2.imwrite(out, crop)
            print("saved", out)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

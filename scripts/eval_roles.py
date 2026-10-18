#!/usr/bin/env python
"""Role-level evaluation of a TFLite file on a hive manifest's val split.

    python scripts/eval_roles.py models/students/hive/hive_int8.tflite data/manifests/hive_plus
    python scripts/eval_roles.py models/students/hive/hive_int8.tflite data/manifests/hive_plus --grayscale
    python scripts/eval_roles.py models/tflite/insects_V1_int8.tflite data/manifests/hive --stock insects_V1

Each val image's true role comes from the manifest's classes.json through
scripts/roles.py. The model's role probability is the sum of its classes'
probabilities. For a student, classes.json gives each output's role; with
--stock, each stock label maps to a role by its iNat2021 taxonomy (matched by
genus where iNat2021 lacks the name). Reports species top-1, role accuracy,
per-role recall and precision, threat recall and honeybee false alarms, the
numbers the hive models are judged on. Same 87.5% center crop as
eval_tflite.py.
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_tflite import load_image  # noqa: E402
from labels import ROOT, load_labels  # noqa: E402
from preprocess import CROP_FRAC  # noqa: E402
from roles import ROLES, record_role, role  # noqa: E402
from teacher_labels import read_manifest  # noqa: E402

THREATS = [r for r, (kind, _) in ROLES.items() if kind == "threat"]
PREDATORS = [r for r, (kind, _) in ROLES.items() if kind in ("threat", "predator")]


def output_roles(args, classes):
    if not args.stock:
        return [record_role(c) for c in classes], [c["name"] for c in classes]
    val = json.load(open(os.path.join(args.root, "val.json"), encoding="utf-8"))
    by_name = {c["name"]: c for c in val["categories"]}
    by_genus = {}
    for c in val["categories"]:
        by_genus.setdefault(c["name"].split()[0], c)
    names = [l.split(" (")[0] for l in load_labels(args.stock)]
    out = []
    for n in names:
        c = by_name.get(n) or by_genus.get(n.split()[0])
        out.append(role(n.split()[0], c["family"], c["order"]) if c else "other insect")
    return out, names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("manifest")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "datasets", "inat2021"))
    ap.add_argument("--stock", help="stock model name (insects_V1, ...) when the model is not a hive student")
    ap.add_argument("--grayscale", action="store_true")
    ap.add_argument("--crop-frac", type=float, default=CROP_FRAC)
    ap.add_argument("--report", help="write the full report, confusion included, as JSON")
    args = ap.parse_args()

    classes = json.load(open(os.path.join(args.manifest, "classes.json"), encoding="utf-8"))
    rows = read_manifest(os.path.join(args.manifest, "val.tsv"))
    out_roles, out_names = output_roles(args, classes)
    role_names = list(ROLES)
    M = np.zeros((len(out_roles), len(role_names)), np.float32)
    for k, r in enumerate(out_roles):
        M[k, role_names.index(r)] = 1

    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=args.model, num_threads=8)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    if out["shape"][1] != len(out_roles):
        sys.exit(f"model has {out['shape'][1]} outputs, the label source has {len(out_roles)}")
    size = int(inp["shape"][1])
    scale, zero = out["quantization"]

    confusion, species_ok, species_n = collections.Counter(), 0, 0
    t0 = time.time()
    for _, rel, label in rows:
        x = load_image(os.path.join(args.root, rel), size, args.grayscale, args.crop_frac)
        interp.set_tensor(inp["index"], x.astype(inp["dtype"])[None])
        interp.invoke()
        raw = interp.get_tensor(out["index"])[0].astype(np.float32)  # uint8 would wrap under arithmetic
        p = (raw - zero) * scale if scale else raw
        truth = record_role(classes[label])
        confusion[(truth, role_names[int(np.argmax(p @ M))])] += 1
        name = classes[label]["name"]
        if name in out_names:
            species_n += 1
            species_ok += out_names[int(np.argmax(p))] == name

    n = len(rows)
    count = lambda pred: sum(v for k, v in confusion.items() if pred(*k))  # noqa: E731
    report = {"model": os.path.basename(args.model), "manifest": args.manifest, "images": n,
              "view": "grayscale" if args.grayscale else "color", "crop_frac": args.crop_frac,
              "species_top1": [species_ok, species_n], "role_accuracy": [count(lambda t, p: t == p), n],
              "threat_recall": [count(lambda t, p: t in THREATS and p in THREATS), count(lambda t, p: t in THREATS)],
              "honeybee_false_alarm": [count(lambda t, p: t == "honeybee" and p in THREATS), count(lambda t, p: t == "honeybee")],
              "honeybee_to_any_predator": [count(lambda t, p: t == "honeybee" and p in PREDATORS), count(lambda t, p: t == "honeybee")],
              "roles": {}, "seconds": round(time.time() - t0, 1)}
    for r in role_names:
        actual, predicted = count(lambda t, p: t == r), count(lambda t, p: p == r)
        if actual or predicted:
            report["roles"][r] = {"recall": [confusion[(r, r)], actual], "precision": [confusion[(r, r)], predicted]}
    pct = lambda a: f"{a[0]}/{a[1]} ({100 * a[0] / a[1]:.1f}%)" if a[1] else "n/a"  # noqa: E731
    print(f"{report['model']} on {args.manifest} val ({report['view']}, crop {args.crop_frac}, {n} images)")
    for key in ("species_top1", "role_accuracy", "threat_recall", "honeybee_false_alarm", "honeybee_to_any_predator"):
        print(f"  {key:26} {pct(report[key])}")
    for r, v in report["roles"].items():
        print(f"  {r:14} recall {pct(v['recall']):>20}  precision {pct(v['precision']):>20}")
    if args.report:
        report["confusion"] = {f"{t} -> {p}": v for (t, p), v in sorted(confusion.items())}
        json.dump(report, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()

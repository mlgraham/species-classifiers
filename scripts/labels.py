"""Label helpers shared by the scripts.

Two label sources ship with the project and line up index for index:

* models/tfhub/<name>/labelmap.csv       scientific names, from TF Hub
* models/aiy_vision_bonnet/*_labels.txt  "Latin (Common name)" from the AIY kit

The AIY file is used to add common names when it is present.
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_AIY_FILES = {
    "insects_V1": "mobilenet_v2_192res_1.0_inat_insect_labels.txt",
    "plants_V1": "mobilenet_v2_192res_1.0_inat_plant_labels.txt",
    "birds_V1": "mobilenet_v2_192res_1.0_inat_bird_labels.txt",
}


def load_labels(name="insects_V1"):
    """Return a list indexed by class id, e.g. labels[954] -> 'Danaus plexippus (Monarch)'."""
    path = os.path.join(ROOT, "models", "tfhub", name, "labelmap.csv")
    with open(path, encoding="utf-8") as f:
        rows = {int(r["id"]): r["name"] for r in csv.DictReader(f)}
    labels = [rows[i] for i in range(len(rows))]

    aiy = os.path.join(ROOT, "models", "aiy_vision_bonnet", _AIY_FILES.get(name, ""))
    if os.path.exists(aiy):
        with open(aiy, encoding="utf-8") as f:
            aiy_lines = [line.strip() for line in f]
        if len(aiy_lines) == len(labels):
            for i, line in enumerate(aiy_lines):
                m = re.match(r"^(.*?)\s*\((.*)\)\s*$", line)
                if m and m.group(1) == labels[i]:
                    labels[i] = f"{labels[i]} ({m.group(2)})"
    return labels


def load_finetuned_labels(path):
    """Labels written by finetune.py: one class name per line."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]

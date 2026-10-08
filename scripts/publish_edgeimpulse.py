#!/usr/bin/env python
"""Publish the student models to Edge Impulse as public BYOM projects.

    EI_JWT=... python scripts/publish_edgeimpulse.py mammals arachnids

For each taxon: create a public project, upload the int8 TFLite as a
pretrained model, save it as an image classifier (uint8 0..255 input,
crop resize, the labels from models/students/<taxon>/labels.txt), create a
version and make it public. Prints the public URL.

Auth: EI_JWT (the studio session cookie) or EI_API_KEY in the environment.
"""
import os
import sys
import time

import requests

HOST = "https://studio.edgeimpulse.com/v1/api"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DESCRIPTIONS = {
    "mammals": "246-species mammal classifier, MobileNetV2 1.0 at 160 px, int8. 54.9% top-1 / 80.5% top-5 on iNat2021 val (44.9% in grayscale). Source and recipe: github.com/mlgraham/species-classifiers. One of four species classifiers here (mammals, arachnids, herps, fungi), all public; Google's AIY insect, plant and bird models are in the same GitHub repo.",
    "arachnids": "153-species arachnid classifier (spiders, scorpions, ticks), MobileNetV2 1.0 at 160 px, int8. 62.3% top-1 / 86.4% top-5 on iNat2021 val. Source and recipe: github.com/mlgraham/species-classifiers. One of four species classifiers here (mammals, arachnids, herps, fungi), all public; Google's AIY insect, plant and bird models are in the same GitHub repo.",
    "herps": "483-species reptile and amphibian classifier, MobileNetV2 1.0 at 160 px, int8. 46.0% top-1 / 74.4% top-5 on iNat2021 val (34.4% in grayscale). Source and recipe: github.com/mlgraham/species-classifiers. One of four species classifiers here (mammals, arachnids, herps, fungi), all public; Google's AIY insect, plant and bird models are in the same GitHub repo.",
    "fungi": "341-species fungus classifier, MobileNetV2 1.0 at 160 px, int8. 74.0% top-1 / 93.5% top-5 on iNat2021 val. NOT for edibility decisions. Source and recipe: github.com/mlgraham/species-classifiers. One of four species classifiers here (mammals, arachnids, herps, fungi), all public; Google's AIY insect, plant and bird models are in the same GitHub repo.",
}


def headers():
    if os.environ.get("EI_JWT"):
        return {"x-jwt-token": os.environ["EI_JWT"]}
    return {"x-api-key": os.environ["EI_API_KEY"]}


def call(method, path, **kw):
    r = requests.request(method, f"{HOST}{path}", headers=headers(), timeout=300, **kw)
    r.raise_for_status()
    data = r.json()
    if not data.get("success", True):
        raise SystemExit(f"{method} {path}: {data.get('error')}")
    return data


def wait_job(project_id, job_id, what):
    for _ in range(120):
        s = call("GET", f"/{project_id}/jobs/{job_id}/status")["job"]
        if s.get("finished"):
            if not s.get("finishedSuccessful", True):
                raise SystemExit(f"{what} job failed")
            return
        time.sleep(3)
    raise SystemExit(f"{what} job timed out")


def publish(taxon, pid=None):
    labels = open(os.path.join(ROOT, "models", "students", taxon, "labels.txt"), encoding="utf-8").read().splitlines()
    tflite = os.path.join(ROOT, "models", "students", taxon, f"{taxon}_int8.tflite")

    if pid is None:
        created = call("POST", "/projects/create", json={"projectName": f"species-classifier-{taxon}", "projectVisibility": "public", "createApiKey": False})
        pid = created["id"]
        print(f"{taxon}: project {pid} created")
    call("POST", f"/{pid}", json={"description": DESCRIPTIONS[taxon], "projectVisibility": "public", "publicProjectListed": True})

    with open(tflite, "rb") as f:
        up = call("POST", f"/{pid}/pretrained-model/upload", files={"modelFile": (os.path.basename(tflite), f, "application/octet-stream")},
                  data={"modelFileName": os.path.basename(tflite), "modelFileType": "tflite"})
    wait_job(pid, up["id"], "upload")
    info = call("GET", f"/{pid}/pretrained-model")["model"]
    print(f"{taxon}: uploaded; input {info['inputs'][0]['shape']} {info['inputs'][0]['dataType']}, output {info['outputs'][0]['shape']}")

    call("POST", f"/{pid}/pretrained-model/save", json={
        "input": {"inputType": "image", "inputScaling": "0..255", "resizeMode": "crop"},
        "model": {"modelType": "classification", "labels": labels},
    })
    print(f"{taxon}: saved as image classifier with {len(labels)} labels")

    ver = call("POST", f"/{pid}/jobs/version", json={"description": "v0.1.0, matches GitHub release and Hugging Face", "makePublic": True})
    wait_job(pid, ver["id"], "version")
    versions = call("GET", f"/{pid}/versions")["versions"]
    v = max(versions, key=lambda x: x["id"])
    if not v.get("publicProjectUrl") and not v.get("publicProjectId"):
        pub = call("POST", f"/{pid}/jobs/versions/{v['id']}/make-public")
        wait_job(pid, pub["id"], "make-public")
        v = max(call("GET", f"/{pid}/versions")["versions"], key=lambda x: x["id"])
    url = v.get("publicProjectUrl") or f"https://studio.edgeimpulse.com/public/{v.get('publicProjectId', pid)}/latest"
    print(f"{taxon}: public at {url}")
    return url


if __name__ == "__main__":
    # args: taxon or taxon=existing_project_id
    for arg in sys.argv[1:] or ["mammals", "arachnids"]:
        taxon, _, pid = arg.partition("=")
        publish(taxon, int(pid) if pid else None)

#!/usr/bin/env python
"""Cache BioCLIP 2 soft labels for a manifest, for distillation.

    python scripts/teacher_labels.py data/manifests/mammals train
    python scripts/teacher_labels.py data/manifests/mammals val

Writes <manifest>/teacher_<split>.npz with:
    image_ids  int64 [N]
    logits     float16 [N, C]  (100 * cosine similarity, the usual CLIP scale)

Also prints the teacher's own zero-shot top-1 on the split, which is the
ceiling a distilled student can be judged against. Resumable: already
finished splits are skipped unless --force.

BioCLIP 2 is a ViT-L/14; on the Radeon via MPS this runs at roughly 10
images per second, so train_mini mammals (12,300) takes about 20 minutes.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT, load_finetuned_labels  # noqa: E402

HUB = "hf-hub:imageomics/bioclip-2"


def read_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            image_id, rel, label = line.rstrip("\n").split("\t")
            rows.append((int(image_id), rel, int(label)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("split", choices=["train", "val"])
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "datasets", "inat2021"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int, help="first N images only (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = os.path.join(args.manifest, f"teacher_{args.split}.npz")
    if os.path.exists(out_path) and not args.force and not args.limit:
        print(f"{out_path} exists, skipping (use --force)")
        return

    import open_clip

    labels = load_finetuned_labels(os.path.join(args.manifest, "labels.txt"))
    names = [l.split(" (")[0] for l in labels]
    rows = read_manifest(os.path.join(args.manifest, f"{args.split}.tsv"))
    if args.limit:
        rows = rows[: args.limit]

    model, _, preprocess = open_clip.create_model_and_transforms(HUB)
    tokenizer = open_clip.get_tokenizer(HUB)
    model = model.to(args.device).eval()
    with torch.no_grad():
        text = model.encode_text(tokenizer([f"a photo of {n}." for n in names]).to(args.device))
        text = text / text.norm(dim=-1, keepdim=True)

    ids, logits, correct = [], [], 0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            imgs = torch.stack([preprocess(Image.open(os.path.join(args.root, rel)).convert("RGB")) for _, rel, _ in chunk])
            feats = model.encode_image(imgs.to(args.device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            sims = (100.0 * feats @ text.T).float().cpu()
            correct += int((sims.argmax(1) == torch.tensor([lab for _, _, lab in chunk])).sum())
            ids.extend(i for i, _, _ in chunk)
            logits.append(sims.half().numpy())
            done = start + len(chunk)
            if done % (args.batch * 20) == 0 or done == len(rows):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(rows)}  {rate:.1f} img/s  eta {(len(rows) - done) / max(rate, 1e-6) / 60:.0f} min", flush=True)

    logits = np.concatenate(logits)
    if not args.limit:
        np.savez(out_path, image_ids=np.array(ids, dtype=np.int64), logits=logits)
        print(f"wrote {out_path}")
    print(f"teacher zero-shot top-1 on {args.split}: {correct / len(rows):.4f} ({correct}/{len(rows)}) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

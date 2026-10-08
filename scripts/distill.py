#!/usr/bin/env python
"""Train the MobileNetV2 0.5 / 160 px student in PyTorch on the Radeon.

    python scripts/distill.py data/manifests/mammals --out mammals              # hard labels only
    python scripts/distill.py data/manifests/mammals --out mammals_kd --teacher  # + BioCLIP 2 soft labels
    python scripts/distill.py data/manifests/mammals --out smoke --limit 256 --epochs 1

Student: timm mobilenetv2_050 with TF-style "same" padding, ImageNet init,
so the weights transplant 1:1 into Keras MobileNetV2(alpha=0.5) and go out
through scripts/export_tflite.py with the deployment contract unchanged.
Inputs are scaled to -1..1 exactly as the Keras Rescaling layer does.

Loss: cross-entropy on the iNat label, plus, with --teacher, KL to the
cached BioCLIP 2 logits at temperature T weighted by --alpha.

Saves models/torch/<out>.pt (best val top-1) with labels and config inside,
and prints colour and grayscale val top-1 each epoch so the D2 threshold
can be read straight off the log.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT, load_finetuned_labels  # noqa: E402
from teacher_labels import read_manifest  # noqa: E402

TO_MINUS1_1 = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # x/127.5 - 1, the Keras contract


class ManifestDataset(Dataset):
    def __init__(self, rows, root, transform, teacher=None):
        self.rows, self.root, self.transform = rows, root, transform
        self.teacher = teacher  # dict image_id -> logits (float16 np array) or None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        image_id, rel, label = self.rows[i]
        img = Image.open(os.path.join(self.root, rel)).convert("RGB")
        x = self.transform(img)
        t = torch.from_numpy(self.teacher[image_id].astype(np.float32)) if self.teacher else torch.zeros(0)
        return x, label, t


def transforms(size, train, grayscale_p=0.0, force_gray=False):
    if train:
        ops = [
            T.RandomResizedCrop(size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.3, 0.3, 0.2),
        ]
        if grayscale_p > 0:
            ops.append(T.RandomGrayscale(p=grayscale_p))
    else:
        ops = [T.Resize(int(size * 1.14)), T.CenterCrop(size)]
        if force_gray:
            ops.append(T.Grayscale(num_output_channels=3))
    return T.Compose(ops + [T.ToTensor(), TO_MINUS1_1])


def load_teacher(path):
    z = np.load(path)
    return {int(i): l for i, l in zip(z["image_ids"], z["logits"])}


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "datasets", "inat2021"))
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--width", default="050", help="timm mobilenetv2 width suffix: 035, 050, 075, 100")
    ap.add_argument("--init", help="backbone init from scripts/keras_to_torch.py (models/torch/init_<name>.pt) instead of ImageNet")
    ap.add_argument("--resume", help="warm restart: load every weight, classifier included, from a models/torch/<out>.pt saved by this script; the schedule (warmup + cosine at --lr) starts over")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grayscale-p", type=float, default=0.0, help="random grayscale augmentation probability")
    ap.add_argument("--teacher", action="store_true", help="use <manifest>/teacher_train.npz soft labels")
    ap.add_argument("--alpha", type=float, default=0.7, help="weight of the KD term when --teacher")
    ap.add_argument("--temperature", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="use the first N train rows (smoke test)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    import timm

    labels = load_finetuned_labels(os.path.join(args.manifest, "labels.txt"))
    train_rows = read_manifest(os.path.join(args.manifest, "train.tsv"))
    val_rows = read_manifest(os.path.join(args.manifest, "val.tsv"))
    if args.limit:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: max(args.limit // 4, 32)]
    teacher = load_teacher(os.path.join(args.manifest, "teacher_train.npz")) if args.teacher else None

    mk = lambda rows, tf, t=None: DataLoader(  # noqa: E731
        ManifestDataset(rows, args.root, tf, t), batch_size=args.batch, shuffle=t is not None or tf is None,
        num_workers=args.workers, pin_memory=False, persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        ManifestDataset(train_rows, args.root, transforms(args.size, True, args.grayscale_p), teacher),
        batch_size=args.batch, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True,
    )
    val_loader = mk(val_rows, transforms(args.size, False))
    val_gray_loader = mk(val_rows, transforms(args.size, False, force_gray=True))

    model = timm.create_model(f"mobilenetv2_{args.width}", pretrained=not args.init, num_classes=len(labels),
                              drop_rate=0.2, pad_type="same")
    if args.init:
        init = torch.load(args.init, map_location="cpu")
        if init["width"] != args.width:
            sys.exit(f"--init is width {init['width']}, model is {args.width}")
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = init["bn_eps"]
        missing, unexpected = model.load_state_dict(init["backbone"], strict=False)
        assert not unexpected and all(k.startswith("classifier") for k in missing), (missing, unexpected)
        print(f"backbone initialised from {args.init} ({init['source']}); classifier fresh")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ck["labels"] != labels:
            sys.exit(f"--resume labels differ from {args.manifest}")
        if ck["config"]["width"] != args.width:
            sys.exit(f"--resume is width {ck['config']['width']}, model is {args.width}")
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = ck["bn_eps"]
        model.load_state_dict(ck["state_dict"], strict=True)
        print(f"resumed every weight from {args.resume} (epoch {ck['epoch']}, val top-1 {ck['val_top1']:.4f}); fresh schedule at lr {args.lr}")
    model = model.to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * len(train_loader)
    warmup = len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps))
    )

    out_dir = os.path.join(ROOT, "models", "torch")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.out}.pt")
    best = -1.0
    print(f"{len(labels)} classes, {len(train_rows)} train, {len(val_rows)} val, teacher={'yes' if teacher else 'no'}, device={args.device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, seen, loss_sum = time.time(), 0, 0.0
        for x, y, t in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.0 if teacher else 0.1)
            if teacher:
                Tt = args.temperature
                kd = F.kl_div(F.log_softmax(logits / Tt, 1), F.softmax(t.to(args.device) / Tt, 1), reduction="batchmean") * Tt * Tt
                loss = (1 - args.alpha) * loss + args.alpha * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            seen += len(y)
            loss_sum += float(loss) * len(y)
        top1 = evaluate(model, val_loader, args.device)
        gray = evaluate(model, val_gray_loader, args.device)
        rate = seen / (time.time() - t0)
        print(f"epoch {epoch:3d}  loss {loss_sum / max(seen, 1):.3f}  val top-1 {top1:.4f}  gray {gray:.4f}  {rate:.0f} img/s  lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if top1 > best:
            best = top1
            torch.save({"state_dict": model.state_dict(), "labels": labels, "config": vars(args),
                        "bn_eps": next(m.eps for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)),
                        "val_top1": top1, "val_top1_gray": gray, "epoch": epoch}, out_path)
    print(f"best val top-1 {best:.4f}; saved {out_path}")
    with open(os.path.join(out_dir, f"{args.out}_labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(labels) + "\n")


if __name__ == "__main__":
    main()

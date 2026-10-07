#!/usr/bin/env python
"""Fine-tune the iNaturalist MobileNetV2 backbone on your own photos.

Put images in one folder per class:

    data/datasets/backyard/
        apis_mellifera/      *.jpg
        bombus_impatiens/    *.jpg
        not_an_insect/       *.jpg

then:

    python scripts/finetune.py data/datasets/backyard --out backyard
    python scripts/classify.py --model models/keras/backyard.keras --labels models/keras/backyard_labels.txt photo.jpg

Two stages, both CPU-friendly on an 8-core laptop:

  1. head only      backbone frozen, new softmax head trained from scratch
  2. partial unfreeze  the last --unfreeze-from blocks of the backbone are
                       trained at a low learning rate

With ~200 images per class and a dozen classes expect a few minutes per
epoch on the i9-9980HK. Start with --epochs 5 --epochs2 5 and look at the
validation accuracy before training longer.
"""
import argparse
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import ROOT  # noqa: E402


def datasets(data_dir, size, batch, seed):
    common = dict(image_size=(size, size), batch_size=batch, seed=seed, validation_split=0.2, label_mode="int")
    train = tf.keras.utils.image_dataset_from_directory(data_dir, subset="training", **common)
    val = tf.keras.utils.image_dataset_from_directory(data_dir, subset="validation", **common)
    names = train.class_names
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255)),
        tf.keras.layers.RandomContrast(0.2),
    ])
    train = train.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE), names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--out", required=True, help="name for models/keras/<out>.keras and <out>_labels.txt")
    ap.add_argument("--base", default="insects_V1", help="stock model to start from (insects_V1, plants_V1, birds_V1)")
    ap.add_argument("--epochs", type=int, default=5, help="stage 1 epochs (head only)")
    ap.add_argument("--epochs2", type=int, default=5, help="stage 2 epochs (partial unfreeze), 0 to skip")
    ap.add_argument("--unfreeze-from", default="block_12_expand", help="first backbone layer to train in stage 2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    base_path = os.path.join(ROOT, "models", "keras", f"{args.base}.keras")
    if not os.path.exists(base_path):
        sys.exit(f"{base_path} missing: run scripts/rebuild_keras.py {args.base} first")
    base = tf.keras.models.load_model(base_path)
    backbone = base.get_layer("mobilenetv2_1.00_224")
    size = base.input_shape[1]

    train, val, names = datasets(args.data_dir, size, args.batch, args.seed)
    print(f"{len(names)} classes: {names}")

    inputs = tf.keras.Input(shape=(size, size, 3), name="image")
    x = base.get_layer("rescale")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(names), activation="softmax", name="logits")(x)
    model = tf.keras.Model(inputs, outputs, name=f"inat_finetune_{args.out}")

    out_dir = os.path.join(ROOT, "models", "keras")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(out_dir, f"{args.out}.keras"), save_best_only=True, monitor="val_accuracy"
    )
    with open(os.path.join(out_dir, f"{args.out}_labels.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")

    # stage 1: head only
    backbone.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=[ckpt])

    # stage 2: unfreeze the top of the backbone, keep batch norm frozen
    if args.epochs2 > 0:
        backbone.trainable = True
        seen = False
        for layer in backbone.layers:
            if layer.name == args.unfreeze_from:
                seen = True
            layer.trainable = seen and not isinstance(layer, tf.keras.layers.BatchNormalization)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr2),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train, validation_data=val, epochs=args.epochs2, callbacks=[ckpt])

    print(f"best model saved to models/keras/{args.out}.keras, labels in models/keras/{args.out}_labels.txt")


if __name__ == "__main__":
    main()

# species-classifiers

Species classifiers small enough for microcontrollers and the Coral Edge
TPU: Google's three iNaturalist models from the discontinued AIY Vision Kit,
rescued as trainable weights, plus two new ones trained here on the same
recipe for the taxa Google never shipped. All five are plain MobileNetV2
with a full-integer int8 export and one deployment contract.

| Model | Classes | Input | int8 file | Where it came from |
|---|---|---|---|---|
| `insects_V1` | 1021 species + background | 224 px | 4.0 MB | Google, iNat2017, [TF Hub](https://tfhub.dev/google/aiy/vision/classifier/insects_V1/1) |
| `plants_V1` | 2101 species + background | 224 px | 5.4 MB | Google, iNat2017, [TF Hub](https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1) |
| `birds_V1` | 964 species + background | 224 px | 4.0 MB | Google, iNat2017, [TF Hub](https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1) |
| `mammals` | 246 species | 160 px | 3.0 MB | trained here on iNat2021, `models/students/mammals/` |
| `arachnids` | 153 species | 160 px | 2.9 MB | trained here on iNat2021, `models/students/arachnids/` |

Nothing here needs the Vision Bonnet. Everything runs on a laptop CPU, the
int8 files run as is on anything with TensorFlow Lite Micro, and all five
compile for the Edge TPU with every op mapped.

## Where the weights came from

| File | Source | SHA-256 |
|---|---|---|
| `models/tfhub/insects_V1/saved_model.pb` | [tfhub.dev/google/aiy/vision/classifier/insects_V1/1](https://tfhub.dev/google/aiy/vision/classifier/insects_V1/1) ([Kaggle](https://www.kaggle.com/models/google/aiy/tensorFlow1/vision-classifier-insects-v1)) | `165b6139…6eee8e` |
| `models/tfhub/plants_V1/saved_model.pb` | [tfhub.dev/google/aiy/vision/classifier/plants_V1/1](https://tfhub.dev/google/aiy/vision/classifier/plants_V1/1) | `70aebfb5…a10af` |
| `models/tfhub/birds_V1/saved_model.pb` | [tfhub.dev/google/aiy/vision/classifier/birds_V1/1](https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1) | `7245ed08…14904` |
| `models/tfhub/*/labelmap.csv` | `https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_{insects,plants,birds}_V1_labelmap.csv` | |
| `models/aiy_vision_bonnet/inat_models.tar.gz` | [dl.google.com/dl/aiyprojects/model_exchange/2018-04-13/inat_models.tar.gz](https://dl.google.com/dl/aiyprojects/model_exchange/2018-04-13/inat_models.tar.gz), linked from the [Nature Explorer page](https://aiyprojects.withgoogle.com/model/nature-explorer/) | `19845f3c…9ca7d` |
| `tools/bonnet_model_compiler_latest.tgz` | [dl.google.com/dl/aiyprojects/vision/bonnet_model_compiler_latest.tgz](https://dl.google.com/dl/aiyprojects/vision/bonnet_model_compiler_latest.tgz) | `6b32a02d…7d642` |

Full checksums are in `CHECKSUMS.txt`. The same files also ship in the
`aiy-models` 1.1-0 Debian package at
[dl.google.com/aiyprojects/deb](https://dl.google.com/aiyprojects/deb/pool/main/a/aiy-models/aiy-models_1.1-0_all.deb),
which installs them to `/opt/aiy/models` on the Vision Kit.

The TF Hub files are the trainable weights. The `.binaryproto` files in
`models/aiy_vision_bonnet/` are the same networks compiled for the Vision
Bonnet's Myriad 2 chip. They cannot be converted back, and are kept here only
for reference and for their label files, which carry common names
("Danaus plexippus (Monarch)") that the TF Hub label maps lack. The
`bonnet_model_compiler` is Python 2.7 plus two static Linux x86-64 binaries
and is only useful if a Vision Bonnet ever turns up again.

## Setup

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python scripts/rebuild_keras.py insects_V1     # also plants_V1, birds_V1
python scripts/classify.py data/samples/*.jpg
python scripts/webcam.py
```

`rebuild_keras.py` extracts the frozen constants from the TF1 Hub graph into
a stock `tf.keras.applications.MobileNetV2` with a softmax head and saves
`models/keras/<name>.keras`. It checks itself against the Hub module on the
sample photos. Expect top-1 to agree and probabilities to differ by a few
hundredths, because the Hub graph simulates int8 rounding (it was trained
quantization-aware) and the Keras rebuild runs in float.

Sample output from the stock insect model:

```
data/samples/monarch.jpg
   0.935  Danaus plexippus (Monarch)
data/samples/ladybird.jpg
   0.696  Coccinella septempunctata (Seven-spotted Ladybird)
```

The int8 files run directly, no rebuild needed:

```sh
python scripts/classify.py --model models/students/arachnids/arachnids_int8.tflite \
    --labels models/students/arachnids/labels.txt data/samples/garden_spider.jpg
   0.445  Argiope aurantia (Yellow Garden Spider)
   0.309  Argiope bruennichi (Wasp Spider)
```

## The students

Google only ever published insects, plants, birds and food. The mammal and
arachnid models fill two of the gaps with the same shape of model, so the
same files, scripts and deployment contract apply.

| | mammals | arachnids |
|---|---|---|
| Species | 246 | 153 |
| Training images (iNat2021 train) | 68,917 | 40,687 |
| Validation images (iNat2021 val, 10 per species) | 2,460 | 1,530 |
| int8 top-1, colour | 54.9% | 62.3% |
| int8 top-5, colour | 80.5% | 86.4% |
| int8 top-1, grayscale input | 44.9% | not measured |
| PyTorch checkpoint top-1 | 56.1% | 64.4% |
| Edge TPU | all ops mapped, 3.0 MiB on-chip | all ops mapped, 2.9 MiB on-chip |
| CPU latency, this laptop | 14 ms | 9 ms |

Measured with `scripts/eval_tflite.py` on the int8 file at the 87.5% centre
crop. The files are in `models/students/<taxon>/` with `labels.txt` (one
"Latin name (Common name)" per line, index = class id) and `classes.json`
(the full taxonomy per class, for roll-ups). They are also on Hugging Face
with model cards, hash-identical to the
[v0.1.0 release](https://github.com/mlgraham/species-classifiers/releases/tag/v0.1.0):
[species-classifier-mammals](https://huggingface.co/mlgraham/species-classifier-mammals)
and
[species-classifier-arachnids](https://huggingface.co/mlgraham/species-classifier-arachnids).

**Recipe.** MobileNetV2 at full width and 160 px, initialised from timm's
ImageNet weights, trained with hard labels for 12 epochs on the full
iNat2021 training split with random crops, flips, colour jitter and (for
mammals) 20% random grayscale so the model holds up on a monochrome camera.
Trained in PyTorch on the laptop's Radeon through MPS, moved into Keras with
a positional weight transplant that verifies to a millionth, then exported
exactly like the stock models. About 2.5 hours for mammals, 1.5 for
arachnids.

**How they compare with Google's models.** Scored the same way, int8 file on
iNat2021 validation images at the 87.5% crop. The stock models are scored on
the species they share by Latin name with iNat2021 while still choosing among
all their classes, so their task is harder (more classes) and their training
data older (2017):

| Model | Classes | Val images | Top-1 | Top-5 |
|---|---|---|---|---|
| stock `insects_V1` | 1,022 | 9,510 (951 shared species) | 59.1% | 80.6% |
| stock `birds_V1` | 965 | 8,600 (860 shared species) | 53.5% | 77.3% |
| stock `plants_V1` | 2,102 | 19,580 (1,958 shared species) | 53.2% | 76.8% |
| new `arachnids` | 153 | 1,530 | 62.3% | 86.4% |
| new `mammals` | 246 | 2,460 | 54.9% | 80.5% |

The stock int8 files are calibrated on 300 iNat2021 validation images of
their own taxon; calibrating the plant model on a handful of insect photos,
as an early export did, cost it nine points, so give `export_tflite.py` a
representative `--rep-dir`. The students land in the same band as the originals: a few points either
side on top-1, and a little stronger on top-5, with the caveat that they
have fewer classes to choose from. Measure your own with
`scripts/eval_stock_on_inat2021.py` and `scripts/eval_tflite.py`.

**What didn't work, measured on the 50-image-per-species mini split.**
Half width (0.5) topped out at 33.8% on mammals against 40.0% for full width.
Distilling from BioCLIP 2 soft labels cost two points rather than gaining
any. Initialising from the stock insect backbone rather than ImageNet cost six.
The gains all came from data: moving from 50 to 280 images per species took
the same recipe from 40% to 56%. The scripts for every one of those
experiments are in `scripts/`, so re-running them on another taxon is one
command each:

```sh
python scripts/inat_subset.py Reptiles --train-split train --out data/manifests/reptiles
python scripts/distill.py data/manifests/reptiles --out reptiles --width 100 --epochs 12 --grayscale-p 0.2
python scripts/torch_to_keras.py models/torch/reptiles.pt
python scripts/export_tflite.py models/keras/reptiles.keras --rep-dir data/datasets/inat2021/train
python scripts/eval_tflite.py models/tflite/reptiles_int8.tflite data/manifests/reptiles
scripts/edgetpu_compile.sh models/tflite/reptiles_int8.tflite
```

Width was kept at 1.0 deliberately. A 0.5-width model would run live-rate
on an ESP32-S3 and fit the small Ethos-U55 modules, but it gives up about
six points, and the full-width file still runs on an S3 with PSRAM at a
frame every second or two. If a "micro" variant turns out to matter it can
be trained with `--width 050` from the same manifests.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/rebuild_keras.py` | TF1 Hub graph → trainable `.keras` model |
| `scripts/classify.py` | Top-k predictions for image files |
| `scripts/webcam.py` | Live predictions from the laptop camera, `s` saves a frame |
| `scripts/finetune.py` | Two-stage fine-tune on a folder-per-class dataset |
| `scripts/export_tflite.py` | float32 and full-integer int8 TFLite, ready for `edgetpu_compiler` |
| `scripts/eval_tflite.py` | Top-1 and top-5 of a TFLite file on a manifest's val split, the number to quote |
| `scripts/edgetpu_compile.sh` | Edge TPU compiler 16.0 in Docker, from a mirror since Google retired its apt repo |
| `scripts/inat_subset.py` | iNat2021 annotations → per-taxon manifest and label file |
| `scripts/distill.py` | Train a student in PyTorch on MPS (hard labels, optional soft labels) |
| `scripts/torch_to_keras.py` | PyTorch student → Keras, positionally, verified |
| `scripts/keras_to_torch.py` | Stock Keras backbone → PyTorch init (measured worse than ImageNet, kept for experiments) |
| `scripts/teacher_labels.py` | Cache BioCLIP 2 soft labels for a manifest (measured not to help, kept for experiments) |
| `scripts/eval_stock_on_inat2021.py` | Score a stock model on the iNat2021 species it shares |
| `scripts/tflite_runner.py`, `scripts/labels.py` | Shared helpers |

## Project ideas for the i9-9980HK

The CPU is the right tool here. MobileNetV2 at 224px forward-passes in about
30 ms per image on this machine, and fine-tuning runs at tens of images per
second. TensorFlow dropped GPU support for Intel Macs, and the Radeon is not
worth the detour through PyTorch for a model this small. Each idea below is
an evening to a weekend of compute.

1. **Backyard species subset.** Pick the 10 to 30 insects you actually see
   and fine-tune on them. Pull research-grade photos per species from
   iNaturalist with [pyinaturalist](https://pyinaturalist.readthedocs.io/)
   or the [iNaturalist open data](https://github.com/inaturalist/inaturalist-open-data)
   export, 200 to 500 per class, plus a `not_an_insect` class from your own
   webcam captures. `finetune.py` with the defaults gives a usable model in
   under an hour. This is the natural first project because the backbone has
   already seen these species and only the head has to move.

2. **Taxonomy roll-up head with no new data.** The 1021 species map to a few
   dozen families and about ten orders. Build a lookup from species to
   order (GBIF's name backbone resolves the Latin names), then train a small
   head on the frozen 1280-d embedding to predict order or family. You can
   generate the training signal by relabelling any insect photo set, or by
   summing the stock model's probabilities per group as a soft target. The
   result is far more robust for a live camera, where "some kind of bee" is
   more useful than a wrong species.

3. **Moth light trap.** A white sheet, a UV lamp and the webcam pointed at
   it overnight. Save frames on motion, run the insect model, cluster the
   embeddings per night. Moths dominate the iNaturalist insect set, so the
   stock model is already strong here. Fine-tune on your own trap frames
   after a few nights to handle the lighting.

4. **Pollinator visit counter.** Fine-tune a 3-class model (bee, hoverfly,
   other) and log visits to a flower patch over a day from a webcam clip.
   Small dataset, immediate feedback and it's the kind of model the Coral
   board can run unattended.

5. **Grayscale-robust model for the Dev Board Micro.** The Micro's camera is
   monochrome, so the stock RGB model loses accuracy there. Add a random
   grayscale augmentation in `finetune.py` (or train on grayscale replicated
   to three channels) and compare int8 accuracy on held-out photos before
   and after. Cheap to run, and it directly improves the on-device result.

6. **Another taxon.** iNat2021 has reptiles (313 species), amphibians (170),
   fungi (341), fish, mollusks. The student recipe above is one command per
   step and about two hours of GPU per taxon; see "The students".

7. **Open-set rejection.** The stock `background` class is weak. Collect a
   few thousand non-insect frames from the webcam, fit a simple
   distance-to-centroid or logistic head on the embeddings, and reject
   frames that aren't insects before classification. Pairs well with 2 and 4.

The plant and bird models go through exactly the same scripts with
`--name plants_V1` or `--base birds_V1`.

## Testing on the laptop webcam

```sh
python scripts/webcam.py --name insects_V1
python scripts/webcam.py --model models/keras/backyard.keras --labels models/keras/backyard_labels.txt
```

Hold a phone showing an insect photo in front of the camera. The script
centre-crops the frame, overlays the top three predictions and prints the
inference time. Press `s` to save a crop into `data/captures/`, which is a
quick way to collect a `not_an_insect` class or real-world test images.

## Deployment contract

Everything downstream of this repo consumes one artifact: a full-integer
TFLite file from `scripts/export_tflite.py`. If you are converting it for a
board we don't cover, these are the facts you need.

| | Stock models | Students (mammals, arachnids) and your fine-tunes |
|---|---|---|
| Input tensor | `[1, 224, 224, 3]` uint8, RGB, NHWC | `[1, 160, 160, 3]` uint8, RGB, NHWC (default) |
| Input range | 0 to 255, no normalization. The model rescales to -1..1 internally | same |
| Input quantization | scale 1.0, zero point 0 | same |
| Output tensor | `[1, 1022]` uint8 for insects (1022 = 1021 species + background), `[1, 2102]` plants, `[1, 965]` birds | `[1, 246]` mammals, `[1, 153]` arachnids; `[1, N]` with N = lines in the labels file for your own |
| Output meaning | softmax probabilities, quantized with scale 1/256 and zero point 0, so `prob = value / 256` | same |
| Label index | row `id` in `models/tfhub/<name>/labelmap.csv`; `background` is the last index | line number in `models/students/<taxon>/labels.txt` (or `models/keras/<out>_labels.txt` for your own); no background class |
| Ops | Conv2D, DepthwiseConv2D, Add, AveragePool, FullyConnected, Softmax, Quantize/Dequantize at the edges. Nothing custom, no dynamic shapes | same |
| Float variant | `<stem>_float32.tflite`, float32 in and out, same layout | same |

Centre-crop to a square before resizing; the scripts do this and accuracy
drops if the subject is squashed. For the distilled students an 87.5% centre
crop (resize the short side to 183 px, crop 160) matches how they were
validated and is worth about a point over using the whole frame.

**Targets by tier**

- **Tier 1, the file runs as is.** Anything running TensorFlow Lite Micro or
  LiteRT: ESP32-S3 and P4 via
  [esp-tflite-micro](https://github.com/espressif/esp-tflite-micro), the
  Coral Dev Board Micro's CPU, OpenMV and Arduino Nicla boards, any
  Raspberry Pi, and phones through LiteRT on Android or Core ML on iOS.
- **Tier 2, the file plus the vendor's compiler.** Coral Edge TPU
  (`edgetpu_compiler`, below), Raspberry Pi AI Camera (Sony IMX500
  converter, 8 MB on-sensor limit shared by weights and activations) and
  Arm Ethos-U55 boards such as the Seeed Grove Vision AI V2 (Arm's Vela
  compiler). All three take int8 TFLite in. Most run on x86 Linux only, so
  use Docker on a Mac.
- **Tier 3, a different export.** Luxonis OAK (RVC2 wants FP16 through
  ONNX) and Hailo (its own dataflow compiler). Not covered here.

The reason Tier 1 is free and Tier 2 is one command is that the graph is a
stock MobileNetV2 with a full-integer export. Keep it that way: no custom
layers, no float preprocessing ops in the graph, no dynamic shapes. Every
file here, stock and student, compiles for the Edge TPU with all ops mapped
and parameters fully cached on-chip.

## Deploying to the Coral Dev Board Micro

The Dev Board Micro runs TensorFlow Lite Micro on a Cortex-M7 with an Edge
TPU coprocessor, 64 MB RAM and 128 MB flash. The int8 files are 3 to 5.4 MB,
so every model here fits comfortably. The compiled `_edgetpu.tflite` files
for the two students are already in `models/students/`.

1. Export int8 TFLite with representative images from your dataset:

   ```sh
   python scripts/export_tflite.py models/keras/backyard.keras --rep-dir data/datasets/backyard
   ```

2. Compile for the Edge TPU. Google retired its Coral apt repository (it now
   answers 403), so the helper fetches the unmodified compiler 16.0 package
   from a mirror, checks its hash and runs it in a Debian container. Any
   machine with Docker works:

   ```sh
   scripts/edgetpu_compile.sh models/tflite/backyard_int8.tflite
   ```

   You want every op reported as "Mapped to Edge TPU" and one subgraph. The
   stock models and the full-width students all compile that way, with
   parameters cached on-chip (the stock insect model uses 4.1 MiB of the
   8 MiB cache, the mammal student 3.0 MiB).

3. Drop the `_edgetpu.tflite` into a copy of the
   [coralmicro](https://github.com/google-coral/coralmicro) `classify_images`
   or `camera` example, replace its label file with
   `models/keras/<out>_labels.txt`, build with the coralmicro CMake toolchain
   and flash. The camera delivers 324x324 monochrome frames, so resize to
   224 and replicate the channel. See idea 5 above.

For a Raspberry Pi or any Linux box with a USB Accelerator the same
`_edgetpu.tflite` runs through
[pycoral](https://coral.ai/docs/edgetpu/tflite-python/).

## Notes

- TensorFlow 2.16.2 is the last release with macOS x86_64 wheels, so the
  environment is pinned there with Python 3.11. The same pin has Linux and
  Apple silicon wheels, so the scripts run unchanged elsewhere.
- The Hub modules take 224x224 inputs in the 0 to 1 range and return
  probabilities, not logits. The rebuilt Keras models take 0 to 255 RGB and
  rescale internally, so webcam frames and TFLite int8 inputs need no
  preprocessing.
- The original Vision Bonnet versions ran at 192x192 with fp16 weights, so
  expect the TF Hub versions to be slightly more accurate than the kit was.
- The sample photos are CC0 images from Wikimedia Commons, credited in the
  License section below.

## License

This repository bundles work under several different terms.

**Code** (`scripts/`, `requirements.txt`, this README) is copyright 2026
Michael Graham and licensed under the Apache License 2.0. See `LICENSE`.

**Model weights** in `models/tfhub/` are copyright Google LLC and licensed
under the Apache License 2.0, as stated on their Kaggle model cards. The
rebuilt `.keras` and `.tflite` files you generate from them are derivative
works under the same license. The student models in `models/students/` were
trained here and are released under the Apache License 2.0; they start from
timm's ImageNet weights, also Apache 2.0. See `NOTICE` for the full
attribution.

**Vision Bonnet files** in `models/aiy_vision_bonnet/` and the compiler in
`tools/` were published by Google without stated license terms. They are
kept here for reference and are not covered by the licenses above. If that
matters to you, delete them and use the links in the provenance table.

**Sample photos** in `data/samples/` are CC0 (public domain dedication)
images via Wikimedia Commons:

- `monarch.jpg`: [Danaus plexippus 161367323](https://commons.wikimedia.org/wiki/File:Danaus_plexippus_161367323.jpg)
  by Miranda Kohout, [iNaturalist photo 161367323](https://www.inaturalist.org/photos/161367323).
- `ladybird.jpg`: [Coccinella septempunctata 108856886](https://commons.wikimedia.org/wiki/File:Coccinella_septempunctata_108856886.jpg)
  by Andrey Korobkov, [iNaturalist photo 108856886](https://www.inaturalist.org/photos/108856886).
- `raccoon.jpg`: [Common Raccoon in a Tree](https://commons.wikimedia.org/wiki/File:Common_Raccoon_in_a_Tree.jpg)
  by N0n-Sum-Qua1is-Eram.
- `garden_spider.jpg`: [Argiope aurantia, 2022-08-14, Beechview, 01](https://commons.wikimedia.org/wiki/File:Argiope_aurantia,_2022-08-14,_Beechview,_01.jpg)
  by Cbaile19.
- `dandelion.jpg`: [Taraxacum officinale, 2023-04-14, Beechview, 02](https://commons.wikimedia.org/wiki/File:Taraxacum_officinale,_2023-04-14,_Beechview,_02.jpg)
  by Cbaile19.
- `cardinal.jpg`: [Cardinal in mid flight](https://commons.wikimedia.org/wiki/File:Cardinal_in_mid_flight.jpg)
  by Giigugiigu.

iNaturalist is a joint initiative of the California Academy of Sciences and
the National Geographic Society. This project is not affiliated with
iNaturalist or Google.

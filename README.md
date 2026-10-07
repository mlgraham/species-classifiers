# species-classifiers

Google's iNaturalist species classifiers from the discontinued AIY Vision Kit,
rescued into a form you can run, fine-tune and ship to an Edge TPU from an
Intel MacBook Pro.

Three MobileNetV2 (1.0, 224x224) models trained by Google on the iNaturalist
2017 competition data:

| Model | Classes | What it knows |
|---|---|---|
| `insects_V1` | 1021 species + background | ~1,020 insect species |
| `plants_V1` | 2101 species + background | ~2,100 plant species |
| `birds_V1` | 964 species + background | ~960 bird species |

Nothing here needs the Vision Bonnet. Everything runs on the laptop CPU, and
the export script produces the int8 TFLite file a Coral Edge TPU wants.

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

## Scripts

| Script | Purpose |
|---|---|
| `scripts/rebuild_keras.py` | TF1 Hub graph → trainable `.keras` model |
| `scripts/classify.py` | Top-k predictions for image files |
| `scripts/webcam.py` | Live predictions from the laptop camera, `s` saves a frame |
| `scripts/finetune.py` | Two-stage fine-tune on a folder-per-class dataset |
| `scripts/export_tflite.py` | float32 and full-integer int8 TFLite, ready for `edgetpu_compiler` |
| `scripts/labels.py` | Label loading shared by the above |

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
   Small dataset, immediate feedback, and it's the kind of model the Coral
   board can run unattended.

5. **Grayscale-robust model for the Dev Board Micro.** The Micro's camera is
   monochrome, so the stock RGB model loses accuracy there. Add a random
   grayscale augmentation in `finetune.py` (or train on grayscale replicated
   to three channels) and compare int8 accuracy on held-out photos before
   and after. Cheap to run, and it directly improves the on-device result.

6. **Distil to something smaller.** Train a MobileNetV2 0.35 or 0.5 at 160px
   on the stock model's soft outputs over an unlabelled pile of insect
   photos. Half the latency on the Edge TPU and a fraction of the flash.
   This is the one idea here that benefits from running overnight.

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

## Deploying to the Coral Dev Board Micro

The Dev Board Micro runs TensorFlow Lite Micro on a Cortex-M7 with an Edge
TPU coprocessor, 64 MB RAM and 128 MB flash. The full-integer int8 insect
model is 4.0 MB, so the stock models and any fine-tune fit comfortably.

1. Export int8 TFLite with representative images from your dataset:

   ```sh
   python scripts/export_tflite.py models/keras/backyard.keras --rep-dir data/datasets/backyard
   ```

2. Compile for the Edge TPU. The compiler is Linux x86-64 only. On this
   Intel Mac Docker runs it natively:

   ```sh
   docker run --rm -v "$PWD/models/tflite:/m" -w /m debian:bookworm bash -c '
     apt-get update -qq && apt-get install -y -qq curl gnupg >/dev/null &&
     curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral.gpg &&
     echo "deb [signed-by=/usr/share/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral.list &&
     apt-get update -qq && apt-get install -y -qq edgetpu-compiler >/dev/null &&
     edgetpu_compiler -s backyard_int8.tflite'
   ```

   You want the compiler log to say every op mapped to the Edge TPU. The
   stock models are plain MobileNetV2 and compile fully.

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
- The two sample photos are CC0 iNaturalist observations from Wikimedia
  Commons, credited in the License section below.

## License

This repository bundles work under several different terms.

**Code** (`scripts/`, `requirements.txt`, this README) is copyright 2026
Michael Graham and licensed under the Apache License 2.0. See `LICENSE`.

**Model weights** in `models/tfhub/` are copyright Google LLC and licensed
under the Apache License 2.0, as stated on their Kaggle model cards. The
rebuilt `.keras` and `.tflite` files you generate from them are derivative
works under the same license. See `NOTICE` for the full attribution.

**Vision Bonnet files** in `models/aiy_vision_bonnet/` and the compiler in
`tools/` were published by Google without stated license terms. They are
kept here for reference and are not covered by the licenses above. If that
matters to you, delete them and use the links in the provenance table.

**Sample photos** in `data/samples/` are CC0 (public domain dedication)
iNaturalist observations, via Wikimedia Commons:

- `monarch.jpg`: [Danaus plexippus 161367323](https://commons.wikimedia.org/wiki/File:Danaus_plexippus_161367323.jpg)
  by Miranda Kohout, [iNaturalist photo 161367323](https://www.inaturalist.org/photos/161367323).
- `ladybird.jpg`: [Coccinella septempunctata 108856886](https://commons.wikimedia.org/wiki/File:Coccinella_septempunctata_108856886.jpg)
  by Andrey Korobkov, [iNaturalist photo 108856886](https://www.inaturalist.org/photos/108856886).

iNaturalist is a joint initiative of the California Academy of Sciences and
the National Geographic Society. This project is not affiliated with
iNaturalist or Google.

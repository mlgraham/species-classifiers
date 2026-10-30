---
license: apache-2.0
pipeline_tag: image-classification
library_name: tflite
tags:
  - tflite
  - litert
  - image-classification
  - mobilenet-v2
  - edge-tpu
  - coral
  - esp32
  - microcontroller
  - inaturalist
  - honeybee
  - beekeeping
  - hornet
  - insects
  - species-classification
datasets:
  - inaturalist-2021
base_model: timm/mobilenetv2_100.ra_in1k
---

# Species classifier: hive

A hive guard small enough for microcontrollers and the Coral Edge TPU. Point a camera at a honeybee hive's entrance and it tells the bees from what hunts them: 265 species of bees, wasps, hornets, robber flies, hoverflies, ants and mantises that roll up to 12 roles, among them honeybee, hornet and yellowjacket. MobileNetV2, full-integer int8, trained on iNaturalist photos. Part of [species-classifiers](https://github.com/mlgraham/species-classifiers), which also rescues Google's AIY Vision Kit insect, plant and bird models into the same shape.

Three models with the same classes, one per kind of hardware:

| | `hive` | `hive_coral` | `hive_esp32s3` |
|---|---|---|---|
| Target | every tier: TensorFlow Lite Micro, LiteRT, Edge TPU, Ethos-U55 | Coral Edge TPU (Dev Board Micro, USB Accelerator) | ESP32-S3 with PSRAM (XIAO ESP32S3 Sense) |
| Network | MobileNetV2 1.0, 160 px | MobileNetV2 1.4, 224 px | MobileNetV2 1.0, 128 px |
| int8 file | 3.0 MB | 5.5 MB | 3.1 MB |
| Species top-1 / top-5, color | 71.4% / 90.4% | 80.9% / 94.3% | 66.8% / 88.3% |
| Role accuracy, color | 91.7% | 95.7% | 89.3% |
| Threats caught, color (hornet + yellowjacket) | 321 of 360 (89.2%) | 344 of 360 (95.6%) | 318 of 360 (88.3%) |
| Honeybees called a threat, color | 2 of 60 | 1 of 60 | 1 of 60 |
| Species top-1, grayscale input | 54.1% | 25.4% | 14.6% |
| Threats caught, grayscale | 281 of 360 (78.1%) | 196 of 360 (54.4%) | 144 of 360 (40.0%) |
| Edge TPU | all ops mapped, 3.1 MiB on-chip | all ops mapped, 5.6 MiB on-chip | all ops mapped, 3.1 MiB on-chip |

Measured on the int8 files over 2,650 validation images (10 per species) with an 87.5% center crop. For comparison, Google's stock AIY insect model, rolled up to the same roles, gets 35.7% of roles right and catches 117 of the 360 threats (32.5%).

Use `hive` for a monochrome or infrared-lit camera: it trained with 20% random grayscale. `hive_coral` and `hive_esp32s3` target color cameras and trained on color only.

**What these numbers are not.** One training run per model, picked on the same validation set it is scored on, with 10 images per species and 60 honeybee images. A field false-alarm rate needs footage from a real hive entrance, which is the next test. None of the three has been timed on its board yet; for the ESP32-S3, scaling Espressif's published MobileNetV2 figure by compute gives an estimate of roughly 140 ms a frame.

## Files

| File | What it is |
|---|---|
| `<model>_int8.tflite` | full-integer, uint8 in and out. Runs as is on TensorFlow Lite Micro and LiteRT |
| `<model>_int8_edgetpu.tflite` | the same model compiled with Edge TPU compiler 16.0, every op mapped |
| `<model>_float32.tflite` | float in and out, for desktop use or your own quantization |
| `hive_int8_vela.tflite` | `hive` compiled with Arm's Vela for the Ethos-U55 in the Seeed Grove Vision AI V2, using Seeed's Himax configuration |
| `labels.txt` | one class per line, `Latin name (Common name)`, line number = class index; the same for all three |
| `classes.json` | the full taxonomy per class (kingdom to species) |
| `roles.json` | the role of every output index, and each role's kind (protected, threat, predator, mimic, benign) |

## Roles

| Role | Kind | What it covers |
|---|---|---|
| honeybee | protected | *Apis*, the bees being protected |
| hornet | threat | *Vespa*, which hunt honeybees at the entrance |
| yellowjacket | threat | *Vespula* and *Dolichovespula*, which raid weak hives |
| paper wasp, robber fly, beewolf, mantis, ant | predator | *Polistes* and relatives, Asilidae, *Philanthus*, Mantodea, Formicidae |
| hoverfly | mimic | Syrphidae, bee and wasp mimics |
| bumblebee, other bee, other wasp | benign | *Bombus*, every other bee family, the remaining Hymenoptera |

A role's score is the sum of its species' probabilities and the decision is the highest role score. There is no threshold or timing rule built in, so add your own, for example a threat on several frames in a row.

## Input contract

- Input `[1, S, S, 3]` uint8 RGB, 0 to 255, no normalization (the model rescales internally); S is 160 for `hive`, 224 for `hive_coral` and 128 for `hive_esp32s3`.
- Output `[1, 265]` uint8 probabilities, scale 1/256 and zero point 0, so `prob = value / 256`. No background class.
- Center-crop to a square. The numbers above use an 87.5% center crop (resize the short side to S / 0.875, crop S).
- Ops are plain Conv2D, DepthwiseConv2D, Add, AveragePool, FullyConnected and Softmax.

```python
import json
import numpy as np, tensorflow as tf
from PIL import Image

model, size = "hive_int8.tflite", 160
interp = tf.lite.Interpreter(model_path=model); interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
labels = open("labels.txt").read().splitlines()
roles = json.load(open("roles.json"))["output_roles"]

img = Image.open("photo.jpg").convert("RGB")
w, h = img.size; s = int(min(w, h) * 0.875)
img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size))
interp.set_tensor(inp["index"], np.asarray(img, dtype=np.uint8)[None])
interp.invoke()
probs = interp.get_tensor(out["index"])[0] / 256.0
by_role = {}
for p, r in zip(probs, roles):
    by_role[r] = by_role.get(r, 0.0) + p
print(max(by_role, key=by_role.get), labels[int(np.argmax(probs))])
```

## How it was trained

The 239 species of iNaturalist 2021's Hymenoptera, robber flies, hoverflies and mantises, plus 26 *Vespa*, *Vespula*, *Dolichovespula* and *Apis* species iNat2021 lacks (among them *Vespa mandarinia* and *Vespa orientalis*). 215,799 training images: the iNat2021 train split for those species plus up to 600 CC-licensed research-grade iNaturalist open-data photos per species, with near-duplicates of any validation photo removed. The added species are validated on 10 photos from 10 observations none of whose photos are in training. No region filter.

Initialized from timm's ImageNet weights (`mobilenetv2_100.ra_in1k`, and `mobilenetv2_140.ra_in1k` for `hive_coral`), hard labels, 12 epochs, then a 24-epoch warm restart at learning rate 3e-4 keeping the best epoch on validation (35, 34 and 30 overall), random resized crops, flips and color jitter, AdamW with cosine decay. Trained in PyTorch on an NVIDIA A100 at a university HPC cluster, with access provided by M. Gage Plott; transplanted into Keras with a positional weight copy that verifies to a millionth, then exported with TensorFlow's full-integer converter calibrated on 300 training images. `hive_esp32s3` was picked over a 0.5-width model at 160 px with half its compute, which scored 52.9% species top-1, caught 287 of the 360 threats and called 5 of the 60 honeybees a threat. Every script is in the [GitHub repository](https://github.com/mlgraham/species-classifiers).

## Provenance and license

Weights Apache-2.0, trained for the species-classifiers repository. Initialized from timm's ImageNet weights (Apache-2.0). Training images from the iNaturalist 2021 competition dataset plus CC0 / CC-BY / CC-BY-NC research-grade photos from the iNaturalist open-data collection; species names and taxonomy come from iNaturalist. The same files, with checksums, are in [GitHub release v0.2.0](https://github.com/mlgraham/species-classifiers/releases/tag/v0.2.0).

Also public Edge Impulse projects that deploy straight to supported boards: [hive](https://studio.edgeimpulse.com/public/1152992/latest) and [hive_esp32s3](https://studio.edgeimpulse.com/public/1152994/latest). And on Kaggle Models beside Google's originals: https://www.kaggle.com/models/mgraham0/species-classifiers

Not affiliated with iNaturalist or Google.

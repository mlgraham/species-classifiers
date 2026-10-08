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
  - microcontroller
  - inaturalist
  - reptiles
  - amphibians
  - herpetology
  - species-classification
datasets:
  - inaturalist-2021
base_model: timm/mobilenetv2_100.ra_in1k
---

# Species classifier: herps

A 483-species reptile and amphibian classifier (313 reptiles, 170 amphibians: lizards, snakes, turtles, crocodilians, frogs, salamanders) small enough for microcontrollers and the Coral Edge TPU. MobileNetV2 at full width and 160 px, full-integer int8, trained on the iNaturalist 2021 competition data. Part of [species-classifiers](https://github.com/mlgraham/species-classifiers), which also rescues Google's AIY Vision Kit insect, plant and bird models into the same shape.

| File | What it is |
|---|---|
| `herps_int8.tflite` | 3.3 MB, full-integer, uint8 in and out. Runs as is on TensorFlow Lite Micro (ESP32-S3, Coral Dev Board Micro CPU, Raspberry Pi) and phones through LiteRT |
| `herps_int8_edgetpu.tflite` | the same model compiled with Edge TPU compiler 16.0, every op mapped, 3.37 MiB cached on-chip |
| `herps_float32.tflite` | float in and out, for desktop use or your own quantization |
| `herps_int8_vela.tflite` | the int8 model compiled with Arm's Vela for the Ethos-U55 in the Seeed Grove Vision AI V2, using Seeed's Himax configuration; upload it with SenseCraft AI's Model Assistant |
| `labels.txt` | one class per line, `Latin name (Common name)`, line number = class index |
| `classes.json` | the full taxonomy per class (kingdom to species), for order and family roll-ups |

## Accuracy

Measured on the int8 file over the 4,830 iNaturalist 2021 validation images for these species (10 per species), with an 87.5% centre crop:

| Input | Top-1 | Top-5 |
|---|---|---|
| Colour | 46.0% | 74.4% |
| Grayscale | 34.4% | 62.7% |

The PyTorch checkpoint scores 46.6% top-1. This is the weakest of the models in the collection: reptiles and amphibians share one 483-class head, and the 55% top-1 bar set for it before training was not met, so treat it as a first cut. Top-5 is still useful for narrowing a sighting, and `classes.json` lets you roll predictions up to order (Squamata, Anura, Caudata, Testudines) where it is much more reliable. Grayscale is reported because the model was trained with 20% random grayscale so it holds up on monochrome cameras. For comparison, Google's stock AIY insect model scores 59.1% top-1 on the iNat2021 species it shares, choosing among 1,022 classes.

## Input contract

- Input `[1, 160, 160, 3]` uint8 RGB, 0 to 255, no normalization; the model rescales internally.
- Output `[1, 483]` uint8 probabilities, scale 1/256 and zero point 0, so `prob = value / 256`. No background class.
- Centre-crop to a square. The model was validated with an 87.5% centre crop (resize the short side to 183, crop 160), which is worth about a point over the whole frame.
- Ops are plain Conv2D, DepthwiseConv2D, Add, AveragePool, FullyConnected and Softmax, so the file converts for the Raspberry Pi AI Camera (IMX500) and Arm Ethos-U boards as well as the Edge TPU.

```python
import numpy as np, tensorflow as tf
from PIL import Image

interp = tf.lite.Interpreter(model_path="herps_int8.tflite"); interp.allocate_tensors()
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
labels = open("labels.txt").read().splitlines()

img = Image.open("photo.jpg").convert("RGB")
w, h = img.size; s = min(w, h)
img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((160, 160))
interp.set_tensor(inp["index"], np.asarray(img, dtype=np.uint8)[None])
interp.invoke()
probs = interp.get_tensor(out["index"])[0] / 256.0
for i in np.argsort(-probs)[:3]:
    print(f"{probs[i]:.2f}  {labels[i]}")
```

## How it was trained

MobileNetV2 1.0 at 160 px initialised from timm's `mobilenetv2_100.ra_in1k` ImageNet weights, hard labels, 12 epochs over the full iNat2021 training split (133,082 reptile and amphibian images), random resized crops, flips, colour jitter and 20% random grayscale, AdamW with cosine decay. About six hours in PyTorch on a laptop GPU through MPS, transplanted into Keras with a positional weight copy that verifies to a millionth, then exported with TensorFlow's full-integer converter calibrated on 300 training images. Every script is in the [GitHub repository](https://github.com/mlgraham/species-classifiers); the README's "The students" section lists what was tried and measured not to help.

## Provenance and license

Weights Apache-2.0, trained in the species-classifiers repository. Initialised from timm's ImageNet weights (Apache-2.0). Training images from the iNaturalist 2021 competition dataset; species names and taxonomy in `labels.txt` and `classes.json` come from it. The same files, with checksums, are in [GitHub release v0.1.0](https://github.com/mlgraham/species-classifiers/releases/tag/v0.1.0).

Also on Kaggle Models beside Google's originals: https://www.kaggle.com/models/mgraham0/species-classifiers

Not affiliated with iNaturalist or Google.

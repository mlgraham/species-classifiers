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

Measured on the int8 file over the 4,830 iNaturalist 2021 validation images for these species (10 per species), with an 87.5% center crop:

| Input | Top-1 | Top-5 |
|---|---|---|
| Color | 54.3% | 80.5% |
| Grayscale | 42.0% | 69.5% |

The PyTorch checkpoint scores 54.8% top-1 under the same crop. This is the weakest of the models in the collection: reptiles and amphibians share one 483-class head, and the int8 file remains under the 55% top-1 bar set before training, so treat it as a work in progress. Top-5 is still useful for narrowing a sighting, and `classes.json` lets you roll predictions up to order (Squamata, Anura, Caudata, Testudines) where it is much more reliable. Grayscale is reported because the model was trained with 20% random grayscale so it holds up on monochrome cameras. For comparison, Google's stock AIY insect model scores 59.1% top-1 on the iNat2021 species it shares, choosing among 1,022 classes.

## Input contract

- Input `[1, 160, 160, 3]` uint8 RGB, 0 to 255, no normalization; the model rescales internally.
- Output `[1, 483]` uint8 probabilities, scale 1/256 and zero point 0, so `prob = value / 256`. No background class.
- Center-crop to a square. The model was validated with an 87.5% center crop (resize the short side to 183, crop 160), which is worth 2.8 points of top-1 over the whole frame.
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

MobileNetV2 1.0 at 160 px initialized from timm's `mobilenetv2_100.ra_in1k` ImageNet weights, hard labels, 12 epochs over the full iNat2021 training split (133,082 reptile and amphibian images) plus a 6-epoch warm restart on those rows plus 268,709 CC-licensed iNaturalist open-data photos (401,791 images in all), then a 12-epoch and a 24-epoch warm restart on the same 401,791 images at learning rate 3e-4 (the second keeping its best epoch, 23), random resized crops, flips, color jitter and 20% random grayscale, AdamW with cosine decay. The first two stages ran in PyTorch on a laptop GPU through MPS and the last two on an NVIDIA A100 (run by M. Gage Plott on a university HPC cluster); the result was transplanted into Keras with a positional weight copy that verifies to a millionth, then exported with TensorFlow's full-integer converter calibrated on 300 training images. Every script is in the [GitHub repository](https://github.com/mlgraham/species-classifiers); the README's "The students" section lists what was tried and measured not to help.

## Provenance and license

Weights Apache-2.0, trained in the species-classifiers repository. Initialized from timm's ImageNet weights (Apache-2.0). Training images from the iNaturalist 2021 competition dataset plus CC0 / CC-BY / CC-BY-NC research-grade photos from the iNaturalist open-data collection; species names and taxonomy in `labels.txt` and `classes.json` come from iNaturalist 2021. The same files, with checksums, are in [GitHub release v0.1.0](https://github.com/mlgraham/species-classifiers/releases/tag/v0.1.0).

Also a public Edge Impulse project that deploys straight to supported boards: https://studio.edgeimpulse.com/public/1133229/latest. And on Kaggle Models beside Google's originals: https://www.kaggle.com/models/mgraham0/species-classifiers

Not affiliated with iNaturalist or Google.

Six species classifiers as ready-to-run TFLite files: Google's three AIY Vision Kit iNaturalist models (insects, plants, birds) and three new ones trained on the same recipe (mammals, arachnids, herps). Every model ships as three files plus labels:

- `<model>_int8.tflite`: full-integer, uint8 in and out. Runs as is on TensorFlow Lite Micro boards (ESP32-S3, Coral Dev Board Micro CPU, Raspberry Pi, phones through LiteRT).
- `<model>_int8_edgetpu.tflite`: the same model compiled with Edge TPU compiler 16.0, every op mapped, parameters cached on-chip. For the Coral Dev Board Micro, USB Accelerator and M.2 modules.
- `<model>_float32.tflite`: float in and out, for desktop use or your own quantization.
- `<model>_labels.txt`: one class per line, `Latin name (Common name)`, line number = class index.
- `<model>_int8_vela.tflite` (mammals, arachnids, herps only): the int8 model compiled with Arm's Vela for the Ethos-U55 in the Seeed Grove Vision AI V2, using Seeed's Himax configuration. Upload with SenseCraft AI's Model Assistant.

`SHA256SUMS.txt` covers all of it.

## Input contract

| | Stock models (insects, plants, birds) | Students (mammals, arachnids, herps) |
|---|---|---|
| Input | `[1, 224, 224, 3]` uint8 RGB, 0 to 255, no normalization | `[1, 160, 160, 3]`, same |
| Output | uint8 probabilities, scale 1/256; last class is `background` | uint8 probabilities, scale 1/256; no background class |

Centre-crop to a square first. The students were validated with an 87.5% centre crop (resize the short side to 183, crop 160), which is worth about a point over the whole frame. The README's "Deployment contract" section has the rest.

## Accuracy, measured the same way for all six

int8 file, iNaturalist 2021 validation images. The stock models are scored on the species they share by Latin name with iNat2021 while still choosing among all their classes.

| Model | Classes | Val images | Top-1 | Top-5 |
|---|---|---|---|---|
| insects | 1,022 | 9,510 | 59.1% | 80.6% |
| birds | 965 | 8,600 | 53.5% | 77.3% |
| plants | 2,102 | 19,580 | 53.2% | 76.8% |
| arachnids | 153 | 1,530 | 62.3% | 86.4% |
| mammals | 246 | 2,460 | 54.9% | 80.5% |
| herps | 483 | 4,830 | 46.0% | 74.4% |

Mammals in grayscale: 44.9% top-1, herps 34.4% (both were trained with random grayscale for monochrome cameras). Herps is the weakest of the set: reptiles and amphibians share one 483-class head and it missed the 55% bar set for it before training.

## Provenance

The stock weights are Google's, Apache-2.0, from TF Hub; the students were trained in this repository from timm's ImageNet MobileNetV2 weights on iNat2021 and are Apache-2.0. See the README and NOTICE.

## Also on Hugging Face

The three new models, hash-identical to these assets, with model cards: [species-classifier-mammals](https://huggingface.co/mlgraham/species-classifier-mammals), [species-classifier-arachnids](https://huggingface.co/mlgraham/species-classifier-arachnids) and [species-classifier-herps](https://huggingface.co/mlgraham/species-classifier-herps).
Also on Kaggle Models, beside Google's originals: [mgraham0/species-classifiers](https://www.kaggle.com/models/mgraham0/species-classifiers).

Public Edge Impulse projects that deploy straight to supported boards: [mammals](https://studio.edgeimpulse.com/public/1132640/latest), [arachnids](https://studio.edgeimpulse.com/public/1132643/latest).

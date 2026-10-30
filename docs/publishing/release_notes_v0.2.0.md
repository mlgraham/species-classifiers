Ten species classifiers as ready-to-run TFLite files. New in v0.2.0: three hive guards that tell honeybees from the hornets and yellowjackets that hunt them, for a camera at a hive entrance. Everything from v0.1.0 is here unchanged (same bytes, same hashes): Google's three AIY Vision Kit iNaturalist models (insects, plants, birds) and the mammals, arachnids, herps and fungi students. The fungi model is not for edibility decisions.

Every model ships as three files plus labels:

- `<model>_int8.tflite`: full-integer, uint8 in and out. Runs as is on TensorFlow Lite Micro boards (ESP32-S3, Coral Dev Board Micro CPU, Raspberry Pi, phones through LiteRT).
- `<model>_int8_edgetpu.tflite`: the same model compiled with Edge TPU compiler 16.0, every op mapped, parameters cached on-chip. For the Coral Dev Board Micro, USB Accelerator and M.2 modules.
- `<model>_float32.tflite`: float in and out, for desktop use or your own quantization.
- `<model>_labels.txt`: one class per line, `Latin name (Common name)`, line number = class index.
- `<model>_int8_vela.tflite` (mammals, arachnids, herps, fungi and hive): the int8 model compiled with Arm's Vela for the Ethos-U55 in the Seeed Grove Vision AI V2, using Seeed's Himax configuration. Upload with SenseCraft AI's Model Assistant.
- `hive_roles.json`: the hive role of every class index, shared by all three hive models.

`SHA256SUMS.txt` covers all of it.

## The hive models

265 species of bees, wasps, hornets, robber flies, hoverflies, ants and mantises, rolling up to 12 roles: honeybee (protected), hornet and yellowjacket (threats), paper wasp, robber fly, beewolf, mantis and ant (predators), hoverfly (mimic), and bumblebee, other bee and other wasp (benign). A role's score is the sum of its species' probabilities; the decision is the highest role score, with no threshold or timing rule built in.

| | `hive` | `hive_coral` | `hive_esp32s3` |
|---|---|---|---|
| Target | every tier | Coral Edge TPU | ESP32-S3 with PSRAM (XIAO ESP32S3 Sense) |
| Network | MobileNetV2 1.0, 160 px | MobileNetV2 1.4, 224 px | MobileNetV2 1.0, 128 px |
| Species top-1 / top-5, color | 71.4% / 90.4% | 80.9% / 94.3% | 66.8% / 88.3% |
| Role accuracy, color | 91.7% | 95.7% | 89.3% |
| Threats caught, color | 321 of 360 (89.2%) | 344 of 360 (95.6%) | 318 of 360 (88.3%) |
| Honeybees called a threat, color | 2 of 60 | 1 of 60 | 1 of 60 |
| Species top-1, grayscale | 54.1% | 25.4% | 14.6% |

int8 files over 2,650 validation images (10 per species, 87.5% center crop). Google's stock insect model, rolled up the same way, catches 117 of the 360 threats. `hive` trained with 20% random grayscale and suits a monochrome camera; the other two trained on color for color cameras. Trained on 215,799 images (iNaturalist 2021 plus CC-licensed iNaturalist open-data photos) on an NVIDIA A100 at a university HPC cluster, with access provided by M. Gage Plott. One run per model, picked on the same validation set, not yet field-tested at a hive or timed on its board. The README's "The hive models" section has the rest.

## Input contract

| | Stock models (insects, plants, birds) | Students |
|---|---|---|
| Input | `[1, 224, 224, 3]` uint8 RGB, 0 to 255, no normalization | `[1, 160, 160, 3]`, same; `hive_coral` 224, `hive_esp32s3` 128 |
| Output | uint8 probabilities, scale 1/256; last class is `background` | uint8 probabilities, scale 1/256; no background class |

Center-crop to a square first. The students were validated with an 87.5% center crop (resize the short side to 183, crop 160), which is worth 0.5 to 2.8 points of top-1 over the whole frame depending on the model. The README's "Deployment contract" section has the rest.

## Accuracy of the species models, measured the same way for all seven

int8 file, iNaturalist 2021 validation images. The stock models are scored on the species they share by Latin name with iNat2021 while still choosing among all their classes.

| Model | Classes | Val images | Top-1 | Top-5 |
|---|---|---|---|---|
| insects | 1,022 | 9,510 | 59.1% | 80.6% |
| birds | 965 | 8,600 | 53.5% | 77.3% |
| plants | 2,102 | 19,580 | 53.2% | 76.8% |
| arachnids | 153 | 1,530 | 62.3% | 86.4% |
| mammals | 246 | 2,460 | 54.9% | 80.5% |
| herps | 483 | 4,830 | 54.3% | 80.5% |
| fungi | 341 | 3,410 | 74.0% | 93.5% |

Mammals in grayscale: 44.9% top-1, herps 42.0% (both were trained with random grayscale for monochrome cameras). Herps is the weakest of the set: reptiles and amphibians share one 483-class head, and it remains under the 55% bar set for it before training.

## Provenance

The stock weights are Google's, Apache-2.0, from TF Hub; the students were trained for this repository from timm's ImageNet MobileNetV2 weights and are Apache-2.0. See the README and NOTICE.

## Also on Hugging Face, Kaggle and Edge Impulse

Hash-identical to these assets, with model cards: [species-classifier-hive](https://huggingface.co/mlgraham/species-classifier-hive) (all three hive models), [species-classifier-mammals](https://huggingface.co/mlgraham/species-classifier-mammals), [species-classifier-arachnids](https://huggingface.co/mlgraham/species-classifier-arachnids), [species-classifier-herps](https://huggingface.co/mlgraham/species-classifier-herps) and [species-classifier-fungi](https://huggingface.co/mlgraham/species-classifier-fungi).
On Kaggle Models, beside Google's originals: [mgraham0/species-classifiers](https://www.kaggle.com/models/mgraham0/species-classifiers) (TfLite variations `hive`, `hive-coral` and `hive-esp32s3` added).

Public Edge Impulse projects that deploy straight to supported boards: [hive](https://studio.edgeimpulse.com/public/1152992/latest), [hive_esp32s3](https://studio.edgeimpulse.com/public/1152994/latest), [mammals](https://studio.edgeimpulse.com/public/1132640/latest), [arachnids](https://studio.edgeimpulse.com/public/1132643/latest), [herps](https://studio.edgeimpulse.com/public/1133229/latest), [fungi](https://studio.edgeimpulse.com/public/1133231/latest).

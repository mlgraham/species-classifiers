# Publishing checklist (per new taxon)

What was done for mammals and arachnids on 2026-10-07, in order. Each step's inputs are produced by
`scripts/pipeline_taxon.sh <name> <manifest>`: `models/tflite/<name>_{int8,float32,int8_edgetpu,int8_vela}.tflite`,
`models/torch/<name>_labels.txt`, and the measured numbers in `models/torch/<name>.pipeline.log`.

1. **Gate.** Flip the taxon's gate in the ledger on the `eval_tflite.py` int8 number and the Edge TPU log.
2. **Repo.** `models/students/<name>/`: `<name>_int8.tflite`, `<name>_int8_edgetpu.tflite`, `labels.txt`,
   `classes.json` (from the manifest dir). Add a row to the README's model table, a column to "The students"
   table and a row to the comparison table. Regenerate `CHECKSUMS.txt` (see the shasum line in git history).
3. **GitHub release.** `gh release upload v0.1.0 <name>_int8.tflite <name>_float32.tflite <name>_int8_edgetpu.tflite
   <name>_int8_vela.tflite <name>_labels.txt SHA256SUMS.txt --clobber -R mlgraham/species-classifiers`
   after regenerating `SHA256SUMS.txt` over every asset. Add the model to the release notes.
4. **Hugging Face.** Repo `mlgraham/species-classifier-<name>`: the same files plus `classes.json` and a card
   written from `huggingface_card_mammals.md` with the new numbers. Add it to the collection. Token: `HF_TOKEN`
   in `.env`. Verify sha256 via `model_info(files_metadata=True)`.
5. **Kaggle.** Variation under `mgraham0/species-classifiers`: a folder with the files and a
   `model-instance-metadata.json` like `kaggle_model-instance-metadata_mammals.json` (ownerSlug `mgraham0`,
   `modelInstanceType` `ExternalVariant`). `env -u KAGGLE_API_TOKEN kaggle models instances create -p <dir>`
   with `~/.kaggle/kaggle.json`. Wait a minute, download back, verify hashes.
6. **Edge Impulse.** Add a description to `DESCRIPTIONS` in `scripts/publish_edgeimpulse.py`, then
   `EI_JWT=... python scripts/publish_edgeimpulse.py <name>`. `EI_JWT` is the studio session cookie, caught
   from a headed login; it expires.
7. **SenseCraft.** Hardware-gated: upload `<name>_int8_vela.tflite` with the Model Assistant once a Grove
   Vision AI V2 is connected, then publish from My Models.

Fungi additionally needs a "not for edibility decisions" warning in every card and the README.

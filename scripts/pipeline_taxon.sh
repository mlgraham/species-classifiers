#!/usr/bin/env bash
# Train, export, measure and compile one taxon end to end, unattended.
#
#   scripts/pipeline_taxon.sh <name> <manifest-dir> [extra distill.py args...]
#   scripts/pipeline_taxon.sh herps data/manifests/herps --grayscale-p 0.2
#
# Steps, each logged to models/torch/<name>.pipeline.log:
#   1. distill.py        full width, 160 px, ImageNet init, hard labels, 12 epochs
#   2. torch_to_keras.py transplant, verified
#   3. export_tflite.py  float32 + int8, calibrated on 300 images spread over the taxon's train set
#   4. eval_tflite.py    int8 top-1/top-5 on val (colour, and grayscale if --grayscale-p was given)
#   5. edgetpu_compile.sh and vela (Seeed Himax config) on the int8 file
# Publishing (release, Hugging Face, Kaggle, Edge Impulse) is deliberate and separate.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate

NAME="$1"; MANIFEST="$2"; shift 2
LOG="models/torch/${NAME}.pipeline.log"
mkdir -p models/torch models/tflite
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "train ${NAME} from ${MANIFEST} ($*)"
python -u scripts/distill.py "$MANIFEST" --out "$NAME" --width 100 --epochs 12 --batch 128 "$@" 2>&1 \
  | grep --line-buffered -vE "Warning|warn|^$" | tee "models/torch/${NAME}.log" | grep --line-buffered -E "^epoch|^best" | tee -a "$LOG"

say "transplant"
python scripts/torch_to_keras.py "models/torch/${NAME}.pt" 2>&1 | grep -E "saved|verification" | tee -a "$LOG"

say "representative set + export"
REP="/tmp/claude-501/rep_${NAME}"; rm -rf "$REP"; mkdir -p "$REP"
python - "$MANIFEST" "$REP" <<'EOF'
import os, sys
manifest, rep = sys.argv[1:]
rows = [l.split("\t")[1] for l in open(os.path.join(manifest, "train.tsv"))]
step = max(1, len(rows) // 300)
for i, rel in enumerate(rows[::step][:300]):
    os.symlink(os.path.abspath(os.path.join("data/datasets/inat2021", rel)), os.path.join(rep, f"{i:04d}.jpg"))
print("representative images:", len(os.listdir(rep)))
EOF
python scripts/export_tflite.py "models/keras/${NAME}.keras" --rep-dir "$REP" 2>&1 | grep -E "wrote" | tee -a "$LOG"

say "measure int8"
python scripts/eval_tflite.py "models/tflite/${NAME}_int8.tflite" "$MANIFEST" 2>&1 | grep top-1 | tee -a "$LOG"
if [[ " $* " == *" --grayscale-p "* ]]; then
  python scripts/eval_tflite.py "models/tflite/${NAME}_int8.tflite" "$MANIFEST" --grayscale 2>&1 | grep top-1 | tee -a "$LOG"
fi

say "compile"
scripts/edgetpu_compile.sh "models/tflite/${NAME}_int8.tflite" 2>&1 | grep -E "compiled|On-chip memory used|CPU" | tee -a "$LOG"
VELA_CFG="${VELA_CONFIG:-tools/vela_config.ini}"
if [ -f "$VELA_CFG" ]; then
  vela "models/tflite/${NAME}_int8.tflite" --accelerator-config ethos-u55-64 --config "$VELA_CFG" --system-config My_Sys_Cfg --memory-mode My_Mem_Mode_Parent --output-dir models/tflite 2>&1 | grep -E "macs|Flash bandwidth  per" | tee -a "$LOG"
fi
say "done ${NAME}"

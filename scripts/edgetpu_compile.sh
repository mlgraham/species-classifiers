#!/usr/bin/env bash
# Compile int8 TFLite files for the Coral Edge TPU, on any machine with Docker.
#
#   scripts/edgetpu_compile.sh models/tflite/mammals_full_w100_int8.tflite [more.tflite ...]
#
# Google retired the coral-edgetpu-stable apt repo (it answers 403), so this fetches the unmodified
# edgetpu-compiler 16.0 package from the ultralytics/assets mirror, checks its sha256, and runs it in a
# debian:bookworm container (the compiler is a Linux x86-64 binary; Docker on an Intel or Apple-silicon
# Mac both work, the latter under emulation). Output lands next to each input as <name>_edgetpu.tflite
# with a <name>_edgetpu.log listing which ops were mapped. You want every op "Mapped to Edge TPU".
set -euo pipefail

URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/edgetpu-compiler_16.0_amd64.tar.gz"
SHA="f9a97adbf2ff621d24a91721191264f2abc3c1db31e63d9a266d988f907269ce"
CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/edgetpu-compiler-16.0"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <int8.tflite> [...]" >&2
  exit 2
fi
command -v docker >/dev/null || { echo "docker not on PATH; on a Mac start Docker Desktop first (open -a Docker)" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon not running; on a Mac: open -a Docker, then retry" >&2; exit 1; }

if [ ! -x "$CACHE/usr/bin/edgetpu_compiler" ]; then
  mkdir -p "$CACHE"
  curl -sSL -o "$CACHE/compiler.tar.gz" "$URL"
  echo "$SHA  $CACHE/compiler.tar.gz" | shasum -a 256 -c - >/dev/null || { echo "sha256 mismatch on the compiler download" >&2; exit 1; }
  tar xzf "$CACHE/compiler.tar.gz" -C "$CACHE"
fi

for f in "$@"; do
  dir="$(cd "$(dirname "$f")" && pwd)"
  base="$(basename "$f")"
  echo "=== $base ==="
  docker run --rm -v "$CACHE:/compiler:ro" -v "$dir:/m" -w /m debian:bookworm \
    /compiler/usr/bin/edgetpu_compiler -s "$base" | grep -E "compiled|On-chip|Off-chip|subgraphs|Mapped|CPU" || true
done

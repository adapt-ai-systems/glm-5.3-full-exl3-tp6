#!/usr/bin/env bash
# Assemble a docker build context for image/tp4 or image/tp6.
#   image/build-context.sh tp4|tp6 B12X_DIR OUT_DIR
# B12X_DIR = the r17 b12x Python package dir (the `b12x/` package, see image/README.md "b12x"); it is not vendored here.
set -euo pipefail
kind=${1:?tp4|tp6}; b12x=${2:?B12X_DIR}; out=${3:?OUT_DIR}
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/.." && pwd)
[ -f "$b12x/__init__.py" ] || { echo "$b12x is not the b12x package dir" >&2; exit 2; }
mkdir -p "$out"; cp -a "$b12x" "$out/b12x"
cp -a "$here/r17-quantization" "$out/quantization"; cp "$here/patch_runtime.py" "$out/"
case "$kind" in
 tp4) cp "$here/tp4/Dockerfile" "$out/Dockerfile";;
 tp6) cp "$here"/tp6/Dockerfile.* "$out/"
      cp "$repo/overlays/tp4-load-accel/exl3_cached.py" "$out/"
      cp "$repo/overlays/vllm-tp4/sm120_dcp.py" "$repo/overlays/vllm-tp4/mla_attn_dcp.py" "$out/"
      mkdir -p "$out/runtime"; cp "$repo"/runtime/tp6-fragments/*.py "$out/runtime/"
      mkdir -p "$out/e3"; cp -a "$repo/third_party/e3/." "$out/e3/"
      cp "$repo/overlays/vllm-tp6/exl3.py" "$out/exl3-e3.py";;
 *) echo 'tp4|tp6' >&2; exit 2;;
esac
echo "context ready: $out"

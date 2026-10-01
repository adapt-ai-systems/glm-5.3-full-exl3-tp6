#!/bin/bash
# Build libkring.so on a Spark (aarch64) against CUDA 13 CUPTI headers.
# CUDA_TARGET defaults to the host CUDA 13.0 sbsa tree; the rpath also covers a pip CUPTI inside the vLLM image.
set -e
cd "$(dirname "$0")"
C=${CUDA_TARGET:-/usr/local/cuda-13.0/targets/sbsa-linux}
gcc -O2 -fPIC -shared -Wall -o libkring.so kring.c -I$C/include -L$C/lib -lcupti -lpthread \
  -Wl,-rpath,/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:$C/lib:/usr/local/cuda/lib64
echo built $(pwd)/libkring.so

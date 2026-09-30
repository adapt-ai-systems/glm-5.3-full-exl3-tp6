# TP4 image: r17 TR3 EXL3 loader + vendored r17 b12x on the MiaAI ARM base image (see image/README.md for the build context).
FROM ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907
COPY b12x /opt/tr3/b12x
COPY quantization/ /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/
COPY patch_runtime.py /opt/tr3/patch_runtime.py
RUN python3 /opt/tr3/patch_runtime.py && python3 -m compileall -q /opt/tr3 /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
ENV PYTHONPATH=/opt/tr3

# One image, run once per GPU. See deploy/README.md for what containerising
# this does and does not buy — the short version is that it fixes the install,
# not the latency.
#
# The base image pins CUDA 12.6 because the V100 is compute capability 7.0 and
# the default torch wheels ship kernels for 7.5 and up. Building on a newer
# CUDA base and letting pip resolve torch reproduces exactly the failure the
# README documents: every CUDA call dies with "no kernel image is available for
# execution on the device", at runtime, on the target machine only.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# torch first, from the cu126 index, in its own layer. Two reasons: it is 2.5 GB
# and never changes, so it caches; and installing it before anything else is
# what stops pip from pulling the cu13 nvidia-* wheels that torch would then
# silently load in preference to its own.
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install torch==2.13.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

# Verify the trap did not reopen during the build. A container that reports
# +cu130 here will run fine on the build host and fail on every V100.
RUN python3 -c "import torch; v = torch.__version__; print(v); assert 'cu126' in v, v"

COPY app app
COPY scripts scripts
COPY bench bench

# The checkpoint is ~2 GB and is fetched from HuggingFace on first load. Baking
# it into the image would make the image 2 GB larger and pin a version into it;
# mounting a shared cache means every container on the box downloads it once.
ENV HF_HOME=/cache/huggingface

# The archive is mounted read-only. Nothing in the serving path writes to it,
# and a container that cannot write to it cannot corrupt 50 GB that took 18
# minutes to build and a CDS quota to download.
ENV AURORA_DATA_ROOT=/data/global.zarr \
    AURORA_STATIC_FILE=/data/static.zarr \
    AURORA_OUTPUT_DIR=/outputs \
    AURORA_DEVICE=cuda:0 \
    AURORA_AUTOCAST=off

EXPOSE 8000

# One uvicorn worker on purpose. The model instance owns the GPU; a second
# worker in the same container would load a second copy of the checkpoint onto
# the same card. Parallelism is one container per GPU, not one container with
# more workers.
#
# 0.0.0.0 here and 127.0.0.1 when running uvicorn on the host directly, which
# reads like a contradiction and is not. A published port forwards to the
# container's own address, so a server bound to the container's loopback is
# unreachable even from the host. The restriction belongs on the publishing
# side: compose publishes to 127.0.0.1, so this listener is only reachable
# through that. Do not publish these ports on 0.0.0.0 — nothing here
# authenticates. See SECURITY.md.
CMD ["python3", "-m", "uvicorn", "app.api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

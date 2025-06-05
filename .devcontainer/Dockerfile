FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

RUN apt update && apt install -y \
    git curl vim python3 python3-pip python3.10-dev --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

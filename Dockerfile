FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 1) Minimal OS deps & upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates wget build-essential git && \
    rm -rf /var/lib/apt/lists/*

# 2) pip install all the Python stack we need
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir \
    numpy \
    jax[cuda12_pip]>=0.4.20 \
    optax \
    flax \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3) (Optional) Copy your application code here, e.g.:
#    COPY . /app
#    WORKDIR /app

# 4) Entrypoint – container will idle unless overridden by the host
ENTRYPOINT ["sleep", "infinity"]

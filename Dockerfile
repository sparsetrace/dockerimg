# Use CUDA 12.1.1 runtime on Ubuntu 22.04
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 1) Install Python 3.10 + pip + minimal build tools
RUN apt-get update &&     apt-get install -y --no-install-recommends       python3.10       python3.10-distutils       python3-pip       ca-certificates       wget       build-essential       git &&     rm -rf /var/lib/apt/lists/*

#  (Optional) Make "python" point to python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# 2) Upgrade pip & install the Python stack we need
RUN python -m pip install --upgrade pip wheel
RUN python -m pip install --no-cache-dir     numpy     jax[cuda12_pip]>=0.4.20     optax     flax     -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3) (Optional) Copy your application code here, e.g.:
#    COPY . /app
#    WORKDIR /app

# 4) Entrypoint – the container will idle unless overridden by the host
ENTRYPOINT ["sleep", "infinity"]

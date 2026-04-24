#!/bin/bash

FA_VERSION=$1

# assert flash attention version is supported
supported_versions=("2.6.3" "2.7.4" "2.8.3")
if [[ ! " ${supported_versions[@]} " =~ " ${FA_VERSION} " ]]; then
    echo "Unsupported flash attention version: ${FA_VERSION}"
    echo "Supported versions are: ${supported_versions[*]}"
    exit 1
fi

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" | grep -oP '\d+\.\d+')
CUDA_VERSION=$(nvcc --version 2>/dev/null | awk -F'release ' '/release/ {print $2}' | cut -d, -f1 | tr -d '.')
PY_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
AARCH=$(uname -m)

echo "Torch version: $TORCH_VERSION"
echo "CUDA version: $CUDA_VERSION"
echo "Python version: $PY_VERSION"
echo "Architecture: $AARCH"

WHEEL_URL="flash_attn-${FA_VERSION}+cu${CUDA_VERSION}torch${TORCH_VERSION}-cp${PY_VERSION}-cp${PY_VERSION}-linux_${AARCH}.whl"

uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/${WHEEL_URL}

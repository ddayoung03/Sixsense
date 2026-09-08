# source this before running the pipeline:  source env.sh
# Makes TF 2.21 find the pip-installed NVIDIA CUDA/cuDNN libraries in WSL.
VENV="${VENV:-$HOME/sixsense-train/.venv}"
source "$VENV/bin/activate"

_NV="$VENV/lib/python3.12/site-packages/nvidia"
_LIBS=""
for d in "$_NV"/*/lib "$_NV"/*/lib64 "$_NV"/cuda_nvcc/nvvm/lib64; do
    [ -d "$d" ] && _LIBS="$_LIBS:$d"
done
export LD_LIBRARY_PATH="${_LIBS#:}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$_NV/cuda_nvcc/bin:$PATH"          # ptxas for XLA
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$_NV/cuda_nvcc"
export TF_CPP_MIN_LOG_LEVEL=1
unset _NV _LIBS d

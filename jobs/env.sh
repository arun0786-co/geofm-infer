# sourced by the slurm jobs. builds a python env on the node's local disk,
# because the home on ada is small. reused if the node already has it.
set -uo pipefail
if nvidia-smi 2>&1 | grep -qiE "unknown error|no devices were found|requires reset"; then
    echo "gpu on $(hostname) is broken, resubmit with --exclude=$(hostname)"; exit 3
fi
SCR=""
for base in /scratch /ssd_scratch /tmp; do
    if [ -n "$(df -P $base 2>/dev/null | awk 'NR == 2 && $4 > 30 * 1024 * 1024')" ]; then
        SCR=$base/$USER/geofm-infer; break
    fi
done
[ -n "$SCR" ] || { echo "no local disk with 30 GB free on $(hostname)"; exit 3; }
mkdir -p $SCR
export UV_CACHE_DIR=$SCR/cache/uv UV_PYTHON_INSTALL_DIR=$SCR/python HF_HOME=$SCR/hf
if [ ! -x $SCR/tools/bin/uv ]; then
    python3 -m venv $SCR/tools && $SCR/tools/bin/pip install -q uv || exit 1
fi
PY=$SCR/env/bin/python
if [ ! -x $PY ]; then
    $SCR/tools/bin/uv venv -q --python 3.12 $SCR/env || exit 1
fi
$SCR/tools/bin/uv pip install -q --python $PY torch --torch-backend=cu126 || exit 1
$SCR/tools/bin/uv pip install -q --python $PY terratorch rasterio s2cloudless pytest || exit 1
echo "node $(hostname), scratch $SCR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

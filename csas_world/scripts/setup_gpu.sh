#!/usr/bin/env bash
# Source this to run the JAX curling simulator on GPU.
#
# The instance's installed jax_cuda12_plugin (0.9.2) is incompatible with jaxlib
# 0.8.0 and its runtime cuDNN (9.7) is too old. csas_fixed_moreMCTS vendors a
# matching jax/jaxlib/plugin 0.8.0 tree WITH cuDNN >= 9.8 under local_pydeps_jax080.
# Putting that tree first on PYTHONPATH + its nvidia libs on LD_LIBRARY_PATH makes
# JAX use the GPU.
#
# ROOT CAUSE of the recurring "module 'jax_cuda12_plugin.cuda_plugin_extension' has no
# attribute 'ffi_registrations'" crash (2026-06): the vendored jax_cuda12_plugin 0.8.0
# ships WITHOUT an __init__.py -> Python treats it as a PEP-420 *namespace* package, which
# LOSES to the system's 0.9.2 *regular* package (empty __init__.py) regardless of sys.path
# order. So `import jax_cuda12_plugin` grabbed the 0.9.2 extension (which renamed
# ffi_registrations) even with LOCAL_DEPS first. FIX: ensure the vendored dir has an
# __init__.py so it is a regular package and wins the import. We self-heal it below every
# time this is sourced, so it survives the vendored tree being re-extracted/cleaned.
LOCAL_DEPS="${LOCAL_DEPS:-/mnt/data/curling2/csas_fixed_moreMCTS/local_pydeps_jax080}"
PYTORCH_SITE="/opt/pytorch/lib/python3.12/site-packages"
WORLD_SRC="/mnt/data/curling2/csas_world/src"
CSAS_V3_SRC="${CSAS_V3_ROOT:-/mnt/data/curling2/csas_v3}/src"
GPU_LIBS="$(find "${LOCAL_DEPS}/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib 2>/dev/null | paste -sd: -)"

# self-heal: make the vendored cuda plugin a REGULAR package so it shadows the system 0.9.2
if [ -f "${LOCAL_DEPS}/jax_cuda12_plugin/cuda_plugin_extension.so" ] && \
   [ ! -f "${LOCAL_DEPS}/jax_cuda12_plugin/__init__.py" ]; then
    touch "${LOCAL_DEPS}/jax_cuda12_plugin/__init__.py" 2>/dev/null \
        && echo "[setup_gpu] healed: created vendored jax_cuda12_plugin/__init__.py" >&2
fi

export PYTHONPATH="${LOCAL_DEPS}:${PYTORCH_SITE}:${WORLD_SRC}:${CSAS_V3_SRC}:${PYTHONPATH:-}"
# canonical graph-feature env (trunk/prior were trained with this)
export GNN_EDGE_SCALAR_MODE="${GNN_EDGE_SCALAR_MODE:-button_visible_plus_curl_arc_reach_with_outgoing}"
export GNN_NODE_FEATURE_MODE="${GNN_NODE_FEATURE_MODE:-none}"
export GNN_RELEASE_NODE_MODE="${GNN_RELEASE_NODE_MODE:-three_plus_takeout_boundary}"
export GNN_EDGE_PRUNE_MODE="${GNN_EDGE_PRUNE_MODE:-none}"
export LD_LIBRARY_PATH="${GPU_LIBS}:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.30}"
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="/mnt/data/curling2/csas_world/.jax_cache"

# Opt-in GPU self-check: `GPU_VERIFY=1 source scripts/setup_gpu.sh` prints PASS/FAIL loudly so a
# silent CPU fallback (or a real plugin break) can't be mistaken for a working GPU run.
if [ "${GPU_VERIFY:-0}" = "1" ]; then
    python3 - <<'PY' >&2
import sys
try:
    import jax
    devs = jax.devices()
    ok = any(d.platform == "gpu" for d in devs)
    print(f"[setup_gpu] GPU verify: {'PASS' if ok else 'FAIL'} -> {devs}", file=sys.stderr)
    sys.exit(0 if ok else 1)
except Exception as e:
    print(f"[setup_gpu] GPU verify: FAIL -> {e!r}", file=sys.stderr); sys.exit(1)
PY
fi

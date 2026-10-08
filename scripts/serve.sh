#!/bin/bash
# Serve the models in ./models with oMLX + EXL3.
#   Web chat: http://localhost:8000/admin/chat     OpenAI-compatible API: http://localhost:8000/v1
#
#   scripts/serve.sh [extra `omlx serve` arguments]
#
# Environment (all optional):
#   OMLX_EXL3_HOME  install directory from scripts/install.sh   (default ~/omlx-exl3)
#   MODEL_DIR       one folder per model, named as you want it served (default ./models)
#   OMLX_BASE       oMLX state directory: settings, caches       (default ~/.omlx-exl3)
#   PORT            (default 8000)
# On first run, configs/model_settings.json (MTP depth 3 for Qwen3.8-27B-exl3-SC6) is copied into
# OMLX_BASE; after that, edit OMLX_BASE/model_settings.json or use the dashboard's model settings.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
omlx=${OMLX_EXL3_HOME:-$HOME/omlx-exl3}
models=${MODEL_DIR:-$here/models}
base=${OMLX_BASE:-$HOME/.omlx-exl3}
port=${PORT:-8000}

if [ ! -x "$omlx/.venv/bin/omlx" ]; then
  echo "oMLX with EXL3 not found in $omlx: run scripts/install.sh first (or set OMLX_EXL3_HOME)."
  exit 1
fi
mkdir -p "$base/ssd-cache"
if [ ! -f "$base/model_settings.json" ]; then cp "$here/configs/model_settings.json" "$base/model_settings.json"; fi

# A separate state directory keeps this install apart from any other oMLX; don't repoint the
# machine-wide cluster shim at this interpreter either.
export OMLX_SKIP_CLUSTER_SHIM=1
exec "$omlx/.venv/bin/omlx" serve --base-path "$base" --model-dir "$models" --host 127.0.0.1 --port "$port" \
  --paged-ssd-cache-dir "$base/ssd-cache" --paged-ssd-cache-max-size 20GB --log-level info "$@"

#!/bin/bash
# Install oMLX with EXL3 support: upstream oMLX at a pinned commit + this repository's omlx/exl3
# package, tools and two small hook edits, built into its own virtualenv.
#
#   scripts/install.sh [INSTALL_DIR]          (default: ~/omlx-exl3)
#
# Needs: Apple Silicon Mac, git, Xcode command line tools (Metal compiler), Python 3.12
# (found via `uv python find 3.12`, then python3.12 on PATH; or set PYTHON=/path/to/python3.12).
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
dest=${1:-$HOME/omlx-exl3}
OMLX_REPO=https://github.com/jundot/omlx.git
OMLX_REF=a4048eef24b41d885d593742e32a290f3bdb2300     # oMLX main, 2026-10-07

py=${PYTHON:-}
if [ -z "$py" ] && command -v uv >/dev/null 2>&1; then py=$(uv python find 3.12 2>/dev/null || true); fi
if [ -z "$py" ]; then py=$(command -v python3.12 || true); fi
if [ -z "$py" ]; then
  echo "Python 3.12 not found. Install it (e.g. 'uv python install 3.12' or 'brew install python@3.12')"
  echo "or run with PYTHON=/path/to/python3.12."
  exit 1
fi
if [ -e "$dest" ]; then
  echo "$dest already exists; pass another directory or remove it first."
  exit 1
fi

echo "==> oMLX $OMLX_REF -> $dest"
git clone --quiet "$OMLX_REPO" "$dest"
git -C "$dest" checkout --quiet -b exl3 "$OMLX_REF"

echo "==> adding EXL3 support"
cp -R "$here/src/omlx/exl3" "$dest/omlx/exl3"
cp "$here"/src/tools/exl3_*.py "$dest/tools/"
git -C "$dest" apply "$here/patches/omlx-exl3-hooks.patch"

echo "==> building (Python: $py)"
"$py" -m venv "$dest/.venv"
"$dest/.venv/bin/pip" install --quiet --upgrade pip
"$dest/.venv/bin/pip" install --quiet "setuptools>=68" "setuptools-scm>=8" wheel "cmake>=3.27" \
  "nanobind==2.15.0" "mlx==0.32.2"
# the native-kernel build runs cmake from the venv, so the venv's bin must be on PATH
(cd "$dest" && PATH="$dest/.venv/bin:$PATH" OMLX_WITH_CUSTOM_KERNEL=1 .venv/bin/pip install --quiet \
  --no-build-isolation -e .)

"$dest/.venv/bin/python" - <<'EOF'
import mlx.core as mx
import omlx.exl3  # noqa: F401
from omlx.exl3 import kernels
print(f"==> done: oMLX with EXL3 support (mlx {mx.__version__}); kernels module {kernels.__name__}")
EOF
echo "Next: download a model into ./models and run scripts/serve.sh (see README)."

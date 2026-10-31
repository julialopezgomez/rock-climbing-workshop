#!/usr/bin/env bash
# Microban's Python packages for a robot without internet [Ours]: built on the laptop, installed offline.
#
#   bash scripts/robot_wheelhouse.sh [OUT]          (default OUT: dist/robot_wheels)
#   rsync -av dist/robot_wheels/ <robot>:robot_wheels/
#   ssh <robot> 'sh robot_wheels/install.sh'
#
# The packages are those of third_party/microban/uv.lock (what `make setup` would install with uv sync),
# as aarch64 / CPython 3.12 wheels (the robot: Raspberry Pi / Radxa, Python 3.12 [Microban pyproject]);
# microban_model, a git dependency with its meshes in Git LFS, is built here as a wheel. Packages
# published only as source are built here as wheels when they are pure Python. install.sh installs
# them into ~/microban/.venv with `uv pip install --no-index` (no network), creating the venv if it
# is missing (that needs a Python 3.12 on the robot: uv cannot download one offline).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MICROBAN="${MICROBAN:-$ROOT/third_party/microban}"
OUT="$(mkdir -p "${1:-$ROOT/dist/robot_wheels}" && cd "${1:-$ROOT/dist/robot_wheels}" && pwd)"
PLATFORMS=(manylinux2014_aarch64 manylinux_2_17_aarch64 manylinux_2_28_aarch64 manylinux_2_31_aarch64
  manylinux_2_34_aarch64 manylinux_2_35_aarch64 linux_aarch64)
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[1/4] the locked requirements of $MICROBAN"
(cd "$MICROBAN" && uv export --frozen --no-hashes --no-dev --no-emit-project) \
  | grep -v '^\s*#' | grep -v '^\s*$' > "$TMP/all.txt"
grep -v '@ git+' "$TMP/all.txt" > "$OUT/requirements.txt"
GIT_LINE="$(grep '^microban-model @ git+' "$TMP/all.txt" || true)"

echo "[2/4] microban_model, built from git (with its LFS meshes)"
if [[ -n "$GIT_LINE" ]]; then
  url="$(sed -E 's/^microban-model @ git\+([^@]+)@([0-9a-f]+).*/\1/' <<< "$GIT_LINE")"
  rev="$(sed -E 's/^microban-model @ git\+([^@]+)@([0-9a-f]+).*/\2/' <<< "$GIT_LINE")"
  git clone -q "$url" "$TMP/microban_model"
  git -C "$TMP/microban_model" checkout -q "$rev"
  git -C "$TMP/microban_model" lfs pull
  (cd "$TMP/microban_model" && uv build -q --wheel --out-dir "$OUT")
fi

echo "[3/4] the other packages, as aarch64 / CPython 3.12 wheels"
platform_args=()
for p in "${PLATFORMS[@]}"; do platform_args+=(--platform "$p"); done
missing=()
while IFS= read -r req; do
  if ! uvx -q pip download -q --no-deps --only-binary=:all: --python-version 3.12 --implementation cp \
      --abi cp312 --abi abi3 --abi none "${platform_args[@]}" -d "$OUT" "$req" 2>/dev/null; then
    missing+=("$req")
  fi
done < "$OUT/requirements.txt"
# No binary wheel: a pure-Python source package is built here (py3-none-any runs on the robot).
for req in ${missing[@]+"${missing[@]}"}; do
  if uvx -q pip wheel -q --no-deps -w "$TMP/built" "$req" 2>/dev/null \
      && ls "$TMP"/built/*-none-any.whl >/dev/null 2>&1; then
    mv "$TMP"/built/*-none-any.whl "$OUT/"
  else
    echo "  [WARN] no aarch64 wheel for $req: it has to be on the robot already" >&2
    sed -i "\\|^${req}\$|d" "$OUT/requirements.txt"
  fi
  rm -rf "$TMP/built"
done

[[ -n "$GIT_LINE" ]] && echo "microban-model" >> "$OUT/requirements.txt"

echo "[4/4] install.sh"
cat > "$OUT/install.sh" <<'EOF'
#!/bin/sh
# On the robot: Microban's packages from this folder into ~/microban/.venv, no network.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
UV="$(command -v uv || echo ~/.local/bin/uv)"
cd ~/microban
[ -x .venv/bin/python ] || "$UV" venv --offline --python 3.12 .venv
# --no-deps: exactly the locked set (the lock decides, as uv sync; some aarch64 wheels declare
# other bounds than the ones it was locked with)
"$UV" pip install --offline --no-index --no-deps --find-links "$HERE" --python .venv/bin/python \
  -r "$HERE/requirements.txt"
.venv/bin/python -c "import numpy, onnxruntime, microban_model; print('robot env ok:', microban_model.__file__)"
EOF
chmod +x "$OUT/install.sh"
echo "wheelhouse: $OUT ($(ls "$OUT"/*.whl | wc -l) wheels, $(du -sh "$OUT" | cut -f1))"
echo "  rsync -av $OUT/ <robot>:robot_wheels/ && ssh <robot> 'sh robot_wheels/install.sh'"

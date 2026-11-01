#!/bin/sh
set -eu
tools_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root="${RCR_PROJECT_ROOT:-$(CDPATH= cd -- "$tools_dir/.." && pwd)}"
export RCR_PROJECT_ROOT="$project_root"
python_bin="${RCR_PYTHON:-$project_root/.venv/bin/python}"
if [ ! -x "$python_bin" ] && [ -x "$project_root/third_party/simulator/.venv/bin/python" ]; then
    python_bin="$project_root/third_party/simulator/.venv/bin/python"
fi
if [ ! -x "$python_bin" ]; then
    echo "Python environment not found: run 'uv sync' in $project_root or set RCR_PYTHON" >&2
    exit 1
fi
exec "$python_bin" "$tools_dir/dashboard.py" "$@"

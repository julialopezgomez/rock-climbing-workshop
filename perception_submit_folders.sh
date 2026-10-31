#!/usr/bin/env bash
set -euo pipefail

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for runtime in "$script_root/.venv-perception/bin/python" "$script_root/.perception-core/bin/python"; do
    if [[ -x "$runtime" ]]; then
        exec "$runtime" "$script_root/scripts/perception/submit_folders.py" "$@"
    fi
done
printf '%s\n' "No private perception runtime found. Run python3.12 scripts/perception/setup.py --benchmark first." >&2
exit 2

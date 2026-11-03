#!/usr/bin/env bash
set -euo pipefail

script_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export RCR_IMPROVED_MASKS=1
export RCR_IMPROVED_VOLUMES=1
export RCR_IMPROVED_CLASSIFIER=1
export RCR_ALLOW_CUBE=1
export RCR_FINGER_YAW_SCALE=1
export RCR_FINGER_YAW_POLICY=half
exec "$script_root/perception_submit_folders.sh" "$@" \
    --no-native --retry-partial \
    --engine rcr.perception.challenge.improved_frontend:hold_prediction

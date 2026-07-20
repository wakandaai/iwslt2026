#!/usr/bin/env bash
# Run inference on a single audio file.
# Usage: bash scripts/infer.sh CONFIG CHECKPOINT AUDIO LANGUAGE [TASK]

set -euo pipefail

CONFIG=${1:?Usage: $0 CONFIG CHECKPOINT AUDIO LANGUAGE [TASK]}
CHECKPOINT=${2:?}
AUDIO=${3:?}
LANGUAGE=${4:?}
TASK=${5:-transcribe}

python -m st.inference.generate \
    --config     "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --audio      "$AUDIO" \
    --language   "$LANGUAGE" \
    --task       "$TASK"

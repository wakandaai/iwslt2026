#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J s4_v2_batch
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage4_v2_batch.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage4_v2_batch.err

cd /ocean/projects/cis250145p/tanghang/iwslt2026
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/src
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Same step-numbering hazard as train_stage4_v1_batch.sbatch: always prefer
# stage4_v2_batch's own latest checkpoint if any exist; only fall back to
# stage4_v1_batch's checkpoint_step42500 (this run's real seed) if this run
# hasn't saved one yet. Never compare step numbers across the two dirs.
BEST_DIR=""
BEST_STEP=-1
for d in runs/stage4_v2_batch/checkpoint_step*; do
    [ -d "$d" ] || continue
    [ -f "$d/meta.json" ] && [ -f "$d/llm_full.pt" ] && [ -f "$d/optimizer.pt" ] || continue
    s="${d##*checkpoint_step}"
    case "$s" in (*[!0-9]*|"") continue;; esac
    if [ "$s" -gt "$BEST_STEP" ]; then
        BEST_STEP="$s"
        BEST_DIR="$d"
    fi
done

if [ -n "$BEST_DIR" ]; then
    echo "Resuming from this run's own checkpoint: ${BEST_DIR} (step ${BEST_STEP})"
    RESUME_ARG="--resume_from ${BEST_DIR}"
elif [ -d "runs/stage4_v1_batch/checkpoint_step42500" ]; then
    echo "No stage4_v2_batch checkpoint yet — seeding from stage4_v1_batch/checkpoint_step42500"
    RESUME_ARG="--resume_from runs/stage4_v1_batch/checkpoint_step42500"
else
    echo "No checkpoint found — starting from Stage 4 interactive's seed"
    RESUME_ARG="--resume_from runs/stage4_v1_interactive/checkpoint_step50000"
fi

.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.train_st \
    --config configs/experiment/stage4/stage4_v2_batch.yaml \
    ${RESUME_ARG}

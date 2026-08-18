#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J s4_batch
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage4_v1_batch.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage4_v1_batch.err



cd /ocean/projects/cis250145p/tanghang/iwslt2026
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/src
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# IMPORTANT: stage4_v1_batch and stage4_v1_interactive are two DIFFERENT runs
# with their own independent step counters (stage4_v1_batch resets to step 0
# on resume from stage4_v1_interactive, since unfreezing the encoder changes
# the trainable param groups and optimizer.pt fails to load). Their step
# numbers are NOT comparable — e.g. stage4_v1_interactive/checkpoint_step50000
# is that run's *final* checkpoint (its seed for this run), while
# stage4_v1_batch/checkpoint_step42500 is 85% through THIS run's own 50k-step
# cycle. Picking by raw step number across both dirs would wrongly prefer the
# numerically-larger but far-earlier seed checkpoint over this run's own
# actual progress (confirmed to happen in practice on 2026-08-05 — caught
# before it re-seeded and threw away 42,500 steps of real progress). So:
# ALWAYS prefer stage4_v1_batch's own latest checkpoint if any exist; only
# fall back to the stage4_v1_interactive seed if this run hasn't saved one yet.
BEST_DIR=""
BEST_STEP=-1
for d in runs/stage4_v1_batch/checkpoint_step*; do
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
elif [ -d "runs/stage4_v1_interactive/checkpoint_step50000" ]; then
    echo "No stage4_v1_batch checkpoint yet — starting from Stage 4's seed"
    RESUME_ARG="--resume_from runs/stage4_v1_interactive/checkpoint_step50000"
else
    echo "No checkpoint found — starting from Stage 2's seed"
    RESUME_ARG="--resume_from runs/stage2_v2_interactive/checkpoint_step40000"
fi

.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.train_st \
    --config configs/experiment/stage4/stage4_v1_batch.yaml \
    ${RESUME_ARG}

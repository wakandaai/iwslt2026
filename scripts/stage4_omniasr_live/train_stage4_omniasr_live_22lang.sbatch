#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 24:00:00
#SBATCH -A cis250145p
#SBATCH -J stage4_omniasr_live
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage4_omniasr_live/logs/train_stage4_omniasr_live_22lang.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage4_omniasr_live/logs/train_stage4_omniasr_live_22lang.err

# Stage 4: encoder unfrozen, LoRA continues from Stage 3 via --resume_from.
# Set RESUME_FROM below once Stage 3 has actually run. Unvalidated — see
# configs/experiment/stage4/stage4_omniasr_live_22lang.yaml and
# docs/stage1_omniasr_ctc_guide.md before a long run.

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

RESUME_FROM=""  # e.g. runs/stage3_omniasr_live_22lang/checkpoint_stepNNNN
RESUME_FLAG=""
[ -n "$RESUME_FROM" ] && RESUME_FLAG="--resume_from $RESUME_FROM"

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.train_st \
    --config configs/experiment/stage4/stage4_omniasr_live_22lang.yaml \
    $RESUME_FLAG

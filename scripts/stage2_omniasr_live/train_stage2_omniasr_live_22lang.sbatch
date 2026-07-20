#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 24:00:00
#SBATCH -A cis250145p
#SBATCH -J stage2_omniasr_live
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage2_omniasr_live/logs/train_stage2_omniasr_live_22lang.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage2_omniasr_live/logs/train_stage2_omniasr_live_22lang.err

# Stage 2: projector alignment, frozen encoder computed live. Unvalidated —
# see configs/experiment/stage2/stage2_omniasr_live_22lang.yaml and
# docs/stage1_omniasr_ctc_guide.md before a long run.

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.train_st \
    --config configs/experiment/stage2/stage2_omniasr_live_22lang.yaml

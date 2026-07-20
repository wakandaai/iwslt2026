#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage3_omniasr_live
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage3_omniasr_live/logs/stage3_v1.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage3_omniasr_live/logs/stage3_v1.err

# Stage 3: + LoRA, encoder still frozen. Chains off Stage 2's projector
# output via training.projector_checkpoint in the config — set that once
# Stage 2 has actually run. Unvalidated — see
# configs/experiment/stage3/stage3_v1.yaml and
# docs/stage1_omniasr_ctc_guide.md before a long run.

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.train_st \
    --config configs/experiment/stage3/stage3_v1.yaml

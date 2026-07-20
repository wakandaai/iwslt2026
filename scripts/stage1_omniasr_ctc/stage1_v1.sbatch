#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage1_h100_weighted
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/stage1_v1.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/stage1_v1.err

# Validated H100 run — see configs/experiment/stage1/stage1_v1.yaml and
# docs/stage1_omniasr_ctc_guide.md for full run history.

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_v1.yaml \
    --resume_from /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage1_omniasr_ctc_22lang_v100/encoder_step3500.pt

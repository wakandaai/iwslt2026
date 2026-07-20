#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage1_omniasr_ctc
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/train_stage1_omniasr_ctc_22lang.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/train_stage1_omniasr_ctc_22lang.err

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
# WANDB_MODE=offline removed — confirmed compute nodes have outbound internet
# (job 42273049), so W&B logs directly to the cloud (aura-mt project).

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_omniasr_ctc_22lang.yaml

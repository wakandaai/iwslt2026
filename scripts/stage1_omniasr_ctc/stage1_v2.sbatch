#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage1_v2_ddp2
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/stage1_v2.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/stage1_v2.err

# Fresh Stage 1 run (no --resume_from), 2x H100-80 DDP, 50k steps, 5-cycle
# LR warm restarts. See configs/experiment/stage1/stage1_v2.yaml and
# docs/stage1_omniasr_ctc_guide.md for full rationale.

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_v2.yaml

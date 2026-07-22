#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage1_v2_ddp2
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/stage1_v2.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/stage1_v2.err
#SBATCH --open-mode=append

# Stage 1 v2, 2x H100-80 DDP. First 50k steps completed clean
# (val/wer=0.2988) — this resumes from encoder_step50000.pt for 50k more
# steps (2 more 25k-step LR restart cycles, 100k total). See
# configs/experiment/stage1/stage1_v2.yaml and docs/stage1_omniasr_ctc_guide.md.
# Checkpoint lives in the original iwslt2026 repo, not this one (never copied
# in — see README's Status section).

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_v2.yaml \
    --resume_from /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage1_v2_ddp2/encoder_step50000.pt

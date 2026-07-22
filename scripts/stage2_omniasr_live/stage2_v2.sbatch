#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 24:00:00
#SBATCH -A cis250145p
#SBATCH -J stage2_v2
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage2_omniasr_live/logs/stage2_v2.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage2_omniasr_live/logs/stage2_v2.err

# Stage 2 v2: fresh run (no --resume_from), 2x H100-80 DDP, projector alignment
# against the Stage 1 v2 encoder (encoder_step50000.pt, val/wer=0.2988 — the
# fixed/clean pipeline). Not a continuation of stage2_v1's checkpoint_step9000,
# which was trained against the older Stage 1 v1 encoder's feature space.
# train_st.py just gained DDP support (was single-GPU only) — see
# configs/experiment/stage2/stage2_v2.yaml and docs/stage1_omniasr_ctc_guide.md.
# Checkpoints live in the original iwslt2026 repo, not this one (never copied
# in — see README's Status section).

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.train_st \
    --config configs/experiment/stage2/stage2_v2.yaml

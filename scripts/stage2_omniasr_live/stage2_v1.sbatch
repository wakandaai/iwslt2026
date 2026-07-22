#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 24:00:00
#SBATCH -A cis250145p
#SBATCH -J stage2_v1
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage2_omniasr_live/logs/stage2_v1.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage2_omniasr_live/logs/stage2_v1.err
#SBATCH --open-mode=append

# Stage 2, projector alignment, frozen encoder (live). Resumes from
# checkpoint_step9000 (killed by a node hardware failure, not a code issue —
# val_max_batches fix already confirmed working up to that point). eval/save
# frequency raised 500->2500 to cut validation-generation overhead. See
# configs/experiment/stage2/stage2_v1.yaml and docs/stage1_omniasr_ctc_guide.md.
# Checkpoint lives in the original iwslt2026 repo, not this one (never copied
# in — see README's Status section).

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.train_st \
    --config configs/experiment/stage2/stage2_v1.yaml \
    --resume_from /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage2_v1/checkpoint_step9000

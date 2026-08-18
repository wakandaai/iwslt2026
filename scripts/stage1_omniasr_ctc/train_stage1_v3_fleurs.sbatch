#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:4
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage1_v3_fleurs
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage1_v3_fleurs.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage1_v3_fleurs.err
#SBATCH --open-mode=append

# Stage 1 v3, 4x V100-32 DDP. Resumes from Stage 1 v2's encoder_step50000.pt
# (val/wer=0.2988) for 50k more steps (100k total), now against the
# FLEURS-augmented ASR_INDEX_V5_16k.csv instead of V4. See
# configs/experiment/stage1/stage1_v3_fleurs_v100x4.yaml for the full
# rationale (FLEURS regression root-cause + why V100x4 instead of H100x2).

cd /ocean/projects/cis250145p/tanghang/iwslt2026
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/src
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=4 \
    -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_v3_fleurs_v100x4.yaml \
    --resume_from runs/stage1_v2_ddp2/encoder_step50000.pt

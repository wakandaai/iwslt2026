#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH -t 0:30:00
#SBATCH -A cis250145p
#SBATCH -J smoke_resume_ddp_wt
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/smoke_test_ddp_weighted_resume.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/smoke_test_ddp_weighted_resume.err

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/smoke/stage1_omniasr_ctc_ddp_weighted_resume_smoke.yaml \
    --resume_from /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage1_omniasr_ctc_22lang_v100/encoder_step3500.pt

#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage1_h100_weighted
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/train_stage1_omniasr_ctc_h100_weighted.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/train_stage1_omniasr_ctc_h100_weighted.err

# First real H100 run for Stage 1 — see stage1_omniasr_ctc_h100_weighted.yaml
# for the reasoning behind the batch settings (scaled estimate from V100
# data, not yet empirically validated on H100). Resumes from the cancelled
# V100 job's checkpoint_step3500, now with WeightedLanguageSampler
# (beta_language=0.7) and online W&B logging (WANDB_SILENT=true works around
# a wandb SDK bug in _print_logged_in_message, confirmed via job 42276742).

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
export WANDB_SILENT=true

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1_omniasr_ctc_h100_weighted.yaml \
    --resume_from /ocean/projects/cis250145p/tanghang/iwslt2026/runs/stage1_omniasr_ctc_22lang_v100/encoder_step3500.pt

#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 0:30:00
#SBATCH -A cis250145p
#SBATCH -J smoke_weighted_stage1
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/smoke_test_stage1_v100_weighted.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/smoke_test_stage1_v100_weighted.err

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/smoke/stage1_omniasr_ctc_v100_weighted_smoke.yaml

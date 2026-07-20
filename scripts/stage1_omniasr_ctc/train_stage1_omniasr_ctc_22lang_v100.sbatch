#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -t 48:00:00
#SBATCH -A cis250145p
#SBATCH -J stage1_omniasr_ctc_v100
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/train_stage1_omniasr_ctc_22lang_v100.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/scripts/stage1_omniasr_ctc/logs/train_stage1_omniasr_ctc_22lang_v100.err

# V100-32 variant, using batch settings actually tuned for 32GB
# (stage1_omniasr_ctc_22lang_v100.yaml: max_batch_duration=10/max_batch_size=2/
# grad_accum=32) — avoids both the ~2.5 day GPU-shared queue wait for a long
# h100-80 reservation, and the runaway OOM-churn hit by reusing the H100
# config's larger batch settings unchanged (job 42256920: grad_accum=16 meant
# every "step" needed 16 consecutive successful micro-batches, so frequent
# OOMs compounded into a projected ~2000h ETA). Re-validated with a second
# smoke test (job 42261610) at these exact settings: 10/10 steps, zero OOM,
# ~20-25s/step → ~110-140h projected for the full 20k steps. Checkpoints
# every 500 steps, so this can migrate to H100 later via --resume_from if
# useful.

cd /ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/aura-asr-v1/src
# WANDB_MODE=offline removed — confirmed compute nodes have outbound internet
# (job 42273049), so W&B logs directly to the cloud (aura-mt project) instead
# of needing a manual `wandb sync` after the fact.

/ocean/projects/cis250145p/tanghang/iwslt2026/.envs/omniasr_extract/bin/python -m st.training.pretrain_omniasr_ctc \
    --config configs/experiment/stage1/stage1_omniasr_ctc_22lang_v100.yaml

#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:h100-80:2
#SBATCH -t 8:00:00
#SBATCH -A cis250145p
#SBATCH -J s2_resume
#SBATCH -o /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage2_v2_h100_resume.out
#SBATCH -e /ocean/projects/cis250145p/tanghang/iwslt2026/.envs/train_stage2_v2_h100_resume.err

# Disconnect-proof continuation of the Stage 2 v2 run. A batch job has no shell
# dependency at all — unlike interact/salloc, where an SSH drop tears down the
# whole allocation (nohup does NOT help there: Slurm kills the allocation, not
# just the process).
#
# 8h, not 48h: short jobs backfill into scheduling gaps far more easily, so
# this should start sooner than the 48h job it replaces. grad_accum=4 (not the
# 8 in stage2_v2_h100.yaml) because per-step wall time scales with the
# SEQUENTIAL microbatch count — at the measured 1.28 steps/s, 8 would roughly
# halve throughput and finish far less of the run.
#
# Resume point is resolved at RUNTIME, not submit time: the interactive run is
# still advancing while this sits in the queue, so hardcoding a step would
# throw away everything it did in the meantime. Searches both the interactive
# run's directory and this job's own (so a requeue resumes from itself).

cd /ocean/projects/cis250145p/tanghang/iwslt2026
export PYTHONPATH=/ocean/projects/cis250145p/tanghang/iwslt2026/src
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BEST_DIR=""
BEST_STEP=-1
for d in runs/stage2_v2_interactive/checkpoint_step* runs/stage2_v2_h100_resume/checkpoint_step*; do
    [ -d "$d" ] || continue
    # only accept checkpoints that actually finished writing
    [ -f "$d/meta.json" ] && [ -f "$d/projector.pt" ] || continue
    s="${d##*checkpoint_step}"
    case "$s" in (*[!0-9]*|"") continue;; esac
    if [ "$s" -gt "$BEST_STEP" ]; then
        BEST_STEP="$s"
        BEST_DIR="$d"
    fi
done

if [ -n "$BEST_DIR" ]; then
    echo "Resuming from ${BEST_DIR} (step ${BEST_STEP})"
    RESUME_ARG="--resume_from ${BEST_DIR}"
else
    echo "No checkpoint found — starting from scratch"
    RESUME_ARG=""
fi

.envs/omniasr_extract/bin/torchrun --standalone --nproc_per_node=2 \
    -m st.training.train_st \
    --config configs/experiment/stage2/stage2_v2_h100_resume.yaml \
    ${RESUME_ARG}

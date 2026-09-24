#!/bin/bash
#SBATCH --job-name=picai-unet
#SBATCH --partition=dgx
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module purge
module load singularity/3.10.0
module load cuda/12.9


CASE_INPUT="test"
CASE_OUTPUT="outputs"
SIF="picai_baseline_unet_processor.sif"

mkdir -p "${CASE_OUTPUT}/images/cspca-detection-map"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID:-$$}"
mkdir -p "${MPLCONFIGDIR}"

singularity exec --nv --no-home \
  --bind "${CASE_INPUT}:/input" \
  --bind "${CASE_OUTPUT}:/output" \
  "${SIF}" \
  python3 /opt/algorithm/process.py
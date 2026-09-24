#!/bin/bash
#SBATCH --job-name=picai-unet
#SBATCH --partition=dgx
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module purge
module load singularity/3.10.0 cuda/12.9

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_INPUT="test"
CASE_OUTPUT="outputs"
SIF="picai_baseline_unet_processor.sif"

mkdir -p "${CASE_OUTPUT}/images/cspca-detection-map"

singularity exec --nv --no-home \
  --bind "${CASE_INPUT}:/input" \
  --bind "${CASE_OUTPUT}:/output" \
  "${SIF}" \
  python3 /opt/algorithm/process.py
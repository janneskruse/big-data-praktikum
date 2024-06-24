#!/bin/bash

#SBATCH --time=0:30:00 #2:30:00
#SBATCH --job-name="02_fft_pipeline"
#SBATCH --nodes=4 #3
#SBATCH --tasks-per-node=5 #10
#SBATCH --ntasks=20 #30
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=paula
#SBATCH --mail-user=zt75vipu@studserv.uni-leipzig.de #please indicate your own email here
#SBATCH --mail-type=ALL 
#SBATCH -o "02_fft_pipeline.%j.txt"

# Determine the root directory of the Git repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# Navigate to the root directory
cd "$REPO_ROOT"

# RUN SCRIPT
poetry run python -u code/slurm/02_fft_pipeline.py
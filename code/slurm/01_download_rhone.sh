#!/bin/bash

#SBATCH --time=1-23:00:00  # Tag-Stunden
#SBATCH --job-name="01_download_rhone"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Utilize multiple CPUs for parallel processing
#SBATCH --mem=32G  # Allocate 32GB of RAM
#SBATCH --partition=paul-long
#SBATCH --mail-user=cu19icep@studserv.uni-leipzig.de  # Please indicate your own email here
#SBATCH --mail-type=ALL 
#SBATCH -o "01_download_rhone.%j.txt"  # j for the job id

# Determine the root directory of the Git repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# Navigate to the root directory
cd "$REPO_ROOT"

# RUN SCRIPT
poetry run python code/slurm/01_download_rhone.py

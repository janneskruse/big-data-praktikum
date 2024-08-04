#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' .env | xargs)

#SBATCH --time=1:00:00
#SBATCH --job-name="03_s3_upload"
#SBATCH --nodes=1 #3
#SBATCH --tasks-per-node=10 #10
#SBATCH --ntasks=10 #30
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=paula
#SBATCH --mail-user=${EMAIL} # Uses the email from the .env file
#SBATCH --mail-type=ALL 
#SBATCH -o "03_s3_upload.%j.txt"

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate rhoneCube


# Determine the root directory of the Git repository
REPO_ROOT=$(git rev-parse --show-toplevel)

# Navigate to the root directory
cd "$REPO_ROOT"

# Calculate the total number of CPUs allocated by SLURM for this job
TOTAL_CPUS=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
echo "Total CPUs: $TOTAL_CPUS"

# Pass this number to the Python script, which can use it to set the number of processes in multiprocessing.Pool
#poetry run python -u code/slurm/02_fft_pipeline.py $TOTAL_CPUS
python3 -u code/slurm/03_s3_upload.py $TOTAL_CPUS

# # Print CPU information for debugging
# echo "SLURM job allocated CPUs: $SLURM_CPUS_ON_NODE"
# echo "Multiprocessing CPU count: $(python -c 'import multiprocessing as mp; print(mp.cpu_count())')"

# # RUN SCRIPT
# poetry run python -u code/slurm/02_fft_pipeline.py
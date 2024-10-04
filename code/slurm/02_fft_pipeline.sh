#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name="02_fft_pipeline"
#SBATCH --nodes=3 #3
#SBATCH --tasks-per-node=10 #10
#SBATCH --ntasks=30 #30
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000
#SBATCH --partition=paula
#SBATCH --mail-type=ALL 
#SBATCH -o "02_fft_pipeline.%j.txt"

# ACTIVATE ANACONDA
source /home/sc.uni-leipzig.de/${USER}/.bashrc
source activate rhoneCube

# Calculate the total number of CPUs allocated by SLURM for this job
TOTAL_CPUS=$((SLURM_NTASKS * SLURM_CPUS_PER_TASK))
echo "Total CPUs: $TOTAL_CPUS"

# Pass this number to the Python script, which can use it to set the number of processes in multiprocessing.Pool
python3 -u 02_fft_pipeline.py $TOTAL_CPUS

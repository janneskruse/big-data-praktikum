#!/bin/bash

#SBATCH --time=1-23:00:00  # Tag-Stunden
#SBATCH --job-name="unzip_rhone"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Utilize multiple CPUs for parallel processing
#SBATCH --mem=32G  # Allocate 32GB of RAM
#SBATCH --partition=paul-long
#SBATCH --mail-user=cu19icep@studserv.uni-leipzig.de  # Please indicate your own email here
#SBATCH --mail-type=ALL 
#SBATCH -o "Unzip_Rhone.%j.txt"  # j for the job id

# ACTIVATE ENV
source /home/sc.uni-leipzig.de/le837wmue/Big-Data-Praktikum/lexcube/bin/activate

# RUN SCRIPT


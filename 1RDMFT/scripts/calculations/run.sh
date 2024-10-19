#!/bin/bash
#SBATCH --job-name=DUMMY_NAME
#SBATCH --account=def-ayers
#SBATCH --time=00-00:10
#SBATCH --mem=10G
#SBATCH --cpus-per-task=48
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python/3.11.5
source /home/jerhard/envs/1RMDFT/bin/activate 
python ./H2_Curve.py > out 

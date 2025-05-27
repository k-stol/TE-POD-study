#!/bin/bash --login
#SBATCH -p multicore          # Partition is required. Runs on an AMD Genoa hardware.
#SBATCH -n 1                  # Number of tasks (processes). Default is 1.
#SBATCH -c 168                # Number of CPUs per task. Max 168 for one node.
#SBATCH -t 7-0                # Wallclock limit (days-hours). Required! Max 7-0.
#SBATCH --job-name=bivar_te_study # Optional: Job name for easier tracking

# --- Job Script Starts Here ---
echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

# Load any required modulefiles or activate your Python environment.
module purge
module load mpi/gcc/openmpi/4.1.8-gcc-14.2.0
source .venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the Python script
python POD_study_multivar.py

echo "Job finished at $(date)"
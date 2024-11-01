#!/bin/bash
#SBATCH --job-name=preprocessing               # Job name
#SBATCH --output=stdout1                        # Standard output log
#SBATCH --error=stderr1                         # Standard error log
#SBATCH --ntasks=128                           # Number of tasks (Express Serial limit)
#SBATCH --time=24:00:00                         # Time limit (1 day, adjust if needed)
#SBATCH --partition=genx                       # Express Serial partition
#SBATCH --constraint=rome                      # Use 128-core AMD Rome nodes
#SBATCH --mem=128G                             # Request 128GB of memory
#SBATCH --mail-type=ALL                        # Mail notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=toti010@naver.com          # Email address for notifications

# Change to the working directory
cd $SLURM_SUBMIT_DIR

# Load necessary modules
module load openmpi/4.0.7

# Activate conda environment
source ~/.bashrc
conda activate topo

echo
echo The following nodes will be used to run this program:
echo
srun --ntasks=128 hostname
echo

# Run your program with mpiexec
mpiexec -n 128 python3 generate_cc.py

exit 0

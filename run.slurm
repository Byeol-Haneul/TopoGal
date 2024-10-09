#!/bin/bash
#SBATCH --job-name=run             # Job name
#SBATCH --output=stdout                        # Standard output log
#SBATCH --error=stderr                         # Standard error log
#SBATCH --ntasks=1                             # Number of tasks
#SBATCH --cpus-per-task=48                     # Requested CPUs per Task
#SBATCH --time=120:00:00                       # Time limit (120 hours)
#SBATCH --partition=gpu                        # GPU partition
#SBATCH --gpus=a100-sxm4-40gb:4                # Use A100 GPUs with 40GB memory
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

# Print the nodes used for the job
echo
echo The following nodes will be used to run this program:
echo
srun --ntasks=1 hostname
echo

# Run your program
python3 -m torch.distributed.launch --nproc_per_node=4 tune.py 

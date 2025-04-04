#!/bin/bash
#SBATCH --nodes=1                            # Request 16 nodes
#SBATCH --ntasks=32                          # Total number of tasks (16 tasks per node × 16 nodes)
#SBATCH --ntasks-per-node=32                  # Number of tasks to run on each node
#SBATCH --cpus-per-task=1                     # Number of CPU cores per task (1 per task in this case)
#SBATCH --time=12:00:00                        # Time limit (2 hours)
#SBATCH --partition=preempt                   # Preemptive partition
#SBATCH --qos=preempt                         # Use preemptible QOS
#SBATCH --mem=64G                            # Memory requested
#SBATCH --mail-type=ALL                       
#SBATCH --mail-user=toti010@naver.com         

cd $SLURM_SUBMIT_DIR

module load openmpi/4.0.7

source ~/.bashrc
conda activate base

echo
echo The following nodes will be used to run this program:
echo
srun --ntasks=32 hostname
echo

while true; do sleep 1; done
exit 0

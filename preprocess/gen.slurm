#!/bin/bash
#SBATCH --nodes=8                            # Request 16 nodes
#SBATCH --ntasks=512                          # Total number of tasks (16 tasks per node × 16 nodes)
#SBATCH --ntasks-per-node=64                  # Number of tasks to run on each node
#SBATCH --cpus-per-task=1                     # Number of CPU cores per task (1 per task in this case)
#SBATCH --time=12:00:00                        # Time limit (2 hours)
#SBATCH --partition=preempt                   # Preemptive partition
#SBATCH --qos=preempt                         # Use preemptible QOS
#SBATCH --mem=512G                            # Memory requested
#SBATCH --mail-type=ALL                       
#SBATCH --mail-user=toti010@naver.com         

cd $SLURM_SUBMIT_DIR

module load openmpi/4.0.7

source ~/.bashrc
conda activate base

echo
echo The following nodes will be used to run this program:
echo
srun --ntasks=512 hostname
echo

for r in 0.01 0.015 0.02; do
    export R_LINK=$r  # Make R_LINK available to mpiexec
    echo "Running generate_cc.py with R_LINK=$R_LINK"
    mpiexec -n 512 python3 generate_cc.py
done
'''
for nc in 8000 10000; do
    export NUMCUT=$nc  # Make R_LINK available to mpiexec
    echo "Running generate_cc.py with NUMCUT=$NUMCUT"
    mpiexec -n 512 python3 generate_cc.py
done
'''
exit 0

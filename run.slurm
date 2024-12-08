#!/bin/bash
#SBATCH --job-name=run                         # Job name
#SBATCH --output=stdout11                        # Standard output log
#SBATCH --error=stderr11                         # Standard error log
#SBATCH --nodes=15                              # Request two nodes
#SBATCH --ntasks=15                             # Number of tasks (1 per node)
#SBATCH --cpus-per-task=48                     # Requested CPUs per Task
#SBATCH --time=120:00:00                       # Time limit (120 hours)
#SBATCH --partition=gpupreempt                 # GPU partition
#SBATCH --qos=gpupreempt                 # GPU partition

#SBATCH --gpus=a100-sxm4-80gb:60               # Request 4 GPUs per node
#SBATCH --mem=128G                             # Request 128GB of memory per node
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
echo "The following nodes will be used to run this program:"
srun hostname
echo

# Set up the master address using SLURM's node list
MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_IP=$(hostname -I | awk '{print $1}')

# Debug output to check the master address and node rank
echo "Master address for debugging: $MASTER_ADDR, IP: $MASTER_IP"
#squeue -p gpu | sed -n 's/.*workergpu\([0-9]*\)/\1/p' | sed 's/\([0-9]*\)/'\''\1'\''/g' | paste -sd, - | sed 's/^/[ /;s/$/ ]/'
#squeue -p gpupreempt | sed -n 's/.*workergpu\([0-9]*\)/\1/p' | sed 's/\([0-9]*\)/'\''\1'\''/g' | paste -sd, - | sed 's/^/[ /;s/$/ ]/'

'''
srun torchrun \
  --nproc_per_node=4 \
  --nnodes=15 \
  --rdzv_id="tnn_${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_IP}:12348" \
  tune.py --layerType ClusterTNN


srun torchrun \
  --nproc_per_node=4 \
  --nnodes=15 \
  --rdzv_id="gnn_${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_IP}:12345" \
  tune.py --layerType GNN
'''

srun torchrun \
  --nproc_per_node=4 \
  --nnodes=15 \
  --rdzv_id="tetra_${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_IP}:12346" \
  tune.py --layerType TetraTNN

'''
srun torchrun \
  --nproc_per_node=4 \
  --nnodes=20 \
  --rdzv_id="tnn_${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_IP}:12348" \
  tune.py --layerType All


srun torchrun \
  --nproc_per_node=4 \
  --nnodes=10 \
  --rdzv_id="tnn_${SLURM_JOB_ID}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_IP}:12348" \
  tune.py --layerType TNN
'''
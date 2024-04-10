#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH -C gpu
#SBATCH --account=nstaff
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J dl-test
#SBATCH -o module_job_log_%j.out

config_file=./configs/default.yaml
config="default"
run_num="ddp-module"

# load libs
module load pytorch/2.0.1

## you may also do the following if you want:
# module load conda
# conda activate your_env

# for DDP
export MASTER_ADDR=$(hostname)
cmd="python train_multi_gpu.py --yaml_config=$config_file --config=$config --run_num=$run_num"

set -x
srun -l \
    bash -c "
    source export_DDP_vars.sh
    $cmd
    " 

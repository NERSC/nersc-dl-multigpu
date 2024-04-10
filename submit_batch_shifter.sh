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
#SBATCH --image=nersc/pytorch:ngc-23.07-v0
#SBATCH --module=gpu,nccl-2.18
#SBATCH -o shifter_job_log_%j.out

config_file=./configs/default.yaml
config="default"
run_num="ddp-shifter"

# this is the path to your local env for libs on top of the container
# here we have created a local dir in our ~/.local/perlmutter path
# to mirror what the modules do by default
env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_ngc_23_07_v0

# for DDP
export MASTER_ADDR=$(hostname)

cmd="python train_multi_gpu.py --yaml_config=$config_file --config=$config --run_num=$run_num"

set -x
srun -l shifter --env PYTHONUSERBASE=${env} \
    bash -c "
    source export_DDP_vars.sh
    $cmd
    " 

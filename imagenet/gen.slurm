#!/bin/bash
#SBATCH --job-name=imagenet              		   # name of job
#SBATCH --account=nug@gpu
##SBATCH -C v100-16g 							   # reserving 16 GB GPUs only
#SBATCH --partition=gpu_p2          			   # uncomment for gpu_p2 partition gpu_p2
#SBATCH --ntasks=8					 			   # total number of processes (= number of GPUs here)
##SBATCH --ntasks-per-node=4
#SBATCH --nodes=1                                  # reserving 1 node          
#SBATCH --gres=gpu:8                 			   # number of GPUs (1/4 of GPUs)
##SBATCH --cpus-per-task=10           			   # number of cores per task (1/4 of the 4-GPUs node)
#SBATCH --cpus-per-task=3           			   # number of cores per task (with gpu_p2: 1/8 of the 8-GPUs node)
# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         			   # hyperthreading is deactivated
#SBATCH --time=100:00:00             			   # maximum execution time requested (HH:MM:SS)
#SBATCH --output=logfiles/log.out    # name of output file
#SBATCH --error=logfiles/log.error   # name of error file (here, in common with the output file)
#SBATCH --qos=qos_gpu-t4

# cleans out the modules loaded in interactive and inherited by default 
module purge

# loading of modules
module load pytorch-gpu/py3/1.10.1

# echo of launched commands
set -x

# code execution
python -u main.py --data_dir /gpfsdswork/dataset/imagenet/RawImages --save_dir /gpfsstore/rech/nug/udq92qm/imagenet/ \
		--num_classes 1000 --alpha 2.0 --batch_size 1024
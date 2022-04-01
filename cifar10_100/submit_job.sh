# !/bin/bash

#OAR -l /host=1/gpu_device=1,walltime=12:00:00 

#OAR -O /srv/tempdd/svenkata/logFiles/AlignMixup_CVPR22/cifar100/log_temp.txt
#OAR -E /srv/tempdd/svenkata/logFiles/AlignMixup_CVPR22/cifar100/log_temp.error

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh


echo " got the python script"


module load pytorch/1.10.1-py3.8


EXECUTABLE="main.py --dataset cifar10 --data_dir /srv/tempdd/svenkata/Datasets/data \
					--save_dir /nfs/pyrex/raid6/svenkata/weights/AlignMixup_CVPR22/ --epochs 2000 \
					--alpha 2.0 --num_classes 10 --manualSeed 8492"


echo
echo "=============== RUN ${OAR_JOB_ID} ==============="
echo "Running ..."
python ${EXECUTABLE} $*
echo "Done"

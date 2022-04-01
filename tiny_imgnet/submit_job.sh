# !/bin/bash

#OAR -l /host=1/gpu_device=2,walltime=48:00:00 

#OAR -O /srv/tempdd/svenkata/logFiles/AlignMixup_CVPR22/tiny_imagenet/log.txt
#OAR -E /srv/tempdd/svenkata/logFiles/AlignMixup_CVPR22/tiny_imagenet/log.error

#patch to be aware of "module" inside a job
. /etc/profile.d/modules.sh


echo " got the python script"


module load pytorch/1.10.1-py3.8


EXECUTABLE="main.py  --train_dir /nfs/pyrex/raid6/svenkata/Datasets/tiny-imagenet-200/train \
					--val_dir /nfs/pyrex/raid6/svenkata/Datasets/tiny-imagenet-200/val \
					--save_dir /nfs/pyrex/raid6/svenkata/weights/AlignMixup_CVPR22/tiny_imagenet --epochs 1200 \
 					--alpha 2.0 --num_classes 200 --manualSeed 8492"



echo
echo "=============== RUN ${OAR_JOB_ID} ==============="
echo "Running ..."
python ${EXECUTABLE} $*
echo "Done"

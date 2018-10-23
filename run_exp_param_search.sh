#!/bin/zsh
# For debugging
set -ex
#build=$1
#name_suffix=$2

DATE_STR=$(date +%m%d_%H%M)
PWD=$(pwd)

#waittime=5h

POT_FUNC_LIST=('sq_recip') # 'neg_exp')

for POT_FUNC in ${POT_FUNC_LIST[0]}
do
    for RECIP_S in 0.1 #0.3 1.0 3.0 10.0
    do
        for LR in 0.01 0.02 0.03 0.04 0.05
        do
            SUBDIR="bash_exp_test/LR_$[LR]_$DATE_STR"
            SUBDIR_PATH=$PWD/run_data/$SUBDIR
            mkdir -p $SUBDIR_PATH
            # python train.py --method pairwise --learning_rate $LR
            docker run --runtime=nvidia -u $(id -u):$(id -g) -v $(pwd):/ada_mb --name eval -d tf_models python /ada_mb/eval.py --checkpoint_dir $SUBDIR_PATH --log_dir $SUBDIR_PATH
            sleep 15s
            docker run --runtime=nvidia -u $(id -u):$(id -g) -v $(pwd):/ada_mb -it tf_models python /ada_mb/train.py --method --learning_rate --train_log_dir $SUBDIR_PATH
            docker stop eval 
            #sleep $waittime
    done
done

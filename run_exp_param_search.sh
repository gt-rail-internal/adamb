#!/bin/zsh
build=$1
name_suffix=$2

DATE_STR=$(date +%m%d)

waittime=5h

POT_FUNC_LIST=('sq_recip' 'neg_exp')

for POT_FUNC in ${POT_FUNC_LIST[0]}
do
    for RECIP_S in 0.1 0.3 1.0 3.0 10.0
    do
        python train.py --method pairwise --learning_rate 0.03
        sleep $waittime
    done
done
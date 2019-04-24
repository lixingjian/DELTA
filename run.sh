#!/bin/bash

datadir="/mnt/data/lixingjian/benchmark"

#generate channel weights
for data in Caltech60 Stanford_Dogs CUB_200_2011; do
    param="--data_dir=$datadir/$data --base_model=resnet101 --batch_size=32 --lr_init=0.01 --channel_wei=./config/channel_wei.$data.json"
    python -u channel_eval.py $param
done


#train transfer learning tasks
for((i=1;i<4;i++)); do
for data in Caltech60 Stanford_Dogs CUB_200_2011; do
    alpha=0.01
    if [ "$data" == "Stanford_Dogs" ]; then
        alpha=0.1
    fi
    param="--max_iter=6000 --channel_wei=./config/channel_wei.$data.json --alpha=$alpha --batch_size=64 --base_model=resnet101 --data_dir=$datadir/$data --reg_type=att_fea_map --lr_scheduler=explr"
    python -u train.py $param
done
done

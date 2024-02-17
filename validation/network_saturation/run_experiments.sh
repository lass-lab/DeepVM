#!/bin/bash

TRAIN_NODES=17
REMOTE_NODES=1
EPOCH=30
SAVE=1
PORT=1234
BATCH_SIZE=128
SHARD_SIZE=17 # SHARD_SIZE cannot be larger than TRAIN_NODES
REMOTE_BUFFER_SIZE=2
MODEL_NAME="my_model"

###### DON'T MODIFY THESE ######
IP=$(hostname -I | awk '{print $1}')
NNODES=$((TRAIN_NODES+REMOTE_NODES))
################################

mpirun \
-np $NNODES \
-hostfile hosts \
python3 ./network_saturation_validation.py \
$EPOCH $SAVE \
$TRAIN_NODES $REMOTE_NODES \
$IP $PORT \
--batch_size $BATCH_SIZE \
--shard_size $SHARD_SIZE \
--remote_buffer_size $REMOTE_BUFFER_SIZE \
--model_name $MODEL_NAME \
--file_name_include_datetime False \
--file_save_in_dictionary False \
| tee $MODEL_NAME.result

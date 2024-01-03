#!/bin/bash
IP=$(hostname -I | awk '{print $1}')
# NP = # of TRAINING NODES + # of REMOTE NODES
mpirun \
-np 18 \
-hostfile hosts \
python3 ./training_example.py \
30 1 \
17 1 \
$IP 1234 \
--batch_size 128 \
--shard_size 17 \
--remote_buffer_size 2 \
--model_name my_model \
--file_name_include_datetime False \
--file_save_in_dictionary False \
| tee my_model.result

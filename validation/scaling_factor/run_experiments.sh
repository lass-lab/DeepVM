#!/bin/bash

IP=$(hostname -I | awk '{print $1}')

NODES=(32 28 24 20 16 12 8 4 2 1)
MODELS=("resnet18" "resnet152" "effnetv2l")
MASTER_ADDR="$IP" # master address
MASTER_PORT=27170           # master port
EXPNUM=1
TEMPLATE_FILE="./template"
REPEAT=5
EPOCH=3
STAGE=0
SKIP=-1
for MODEL in ${MODELS[@]}; do
  for NODE in ${NODES[@]}; do
    for ((k=0; k<$REPEAT; k++)); do
      if [ $STAGE -le $SKIP ]; then
        ((STAGE++))
        continue
      fi

      echo "Running training with $NODE node(s) and $MODEL, cifar10 dataset"
      
      for (( i=0; i<$NODE; i++ )); do
        sed "s/{nnodes}/$NODE/g; s/{rank}/$i/g; s/{ip}/$MASTER_ADDR/g; s/{port}/$MASTER_PORT/g; s/{epoch}/$EPOCH/g; s/{model_name}/$MODEL/g; s/{expnum}/$EXPNUM/g" $TEMPLATE_FILE > temp/temp_$i.sh
        chmod +x temp/temp_$i.sh
        ssh worker$((i+1)) bash < temp/temp_$i.sh &
      done
      wait
      
      echo "Training completed"
      MASTER_PORT=$((MASTER_PORT + 1))
      for((i=2;i<=32;i++)); do #backup
        scp ./result_$EXPNUM.txt worker$i:~/ &
      done
      wait
      echo "$EXPNUM exp done."
    done
  done
  EXPNUM=$((EXPNUM + 1))
done

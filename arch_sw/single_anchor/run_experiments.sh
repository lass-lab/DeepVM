#!/bin/bash

NNODES=4
MASTER_PORT=27170 # master port
EPOCH=30
EXPNUM=1
TEMPLATE_FILE="./template"
REPEAT=1
STAGE=0
SKIP=-1

###### DON'T MODIFY THESE ######
IP=$(hostname -I | awk '{print $1}')
MASTER_ADDR="$IP" # master address
################################

for ((k=0; k<$REPEAT; k++)); do
  if [ $STAGE -le $SKIP ]; then
    ((STAGE++))
    continue
  fi

  echo "Running training with $NNODES node(s)"
  
  for (( i=0; i<$NNODES; i++ )); do
    sed "s/{nodes}/$NNODES/g; s/{rank}/$i/g; s/{ip}/$MASTER_ADDR/g; s/{port}/$MASTER_PORT/g; s/{epoch}/$EPOCH/g" $TEMPLATE_FILE > temp/temp_$i.sh # 쉘 생성
    chmod +x temp/temp_$i.sh # 권한 부여
    ssh worker$((i+1)) bash < temp/temp_$i.sh & # 각 노드에서 원격으로 쉘 실행
  done
  wait
  
  echo "Training completed"
  MASTER_PORT=$((MASTER_PORT + 1))
done
EXPNUM=$((EXPNUM + 1))

#!/bin/bash
NNODES=4

python3 ./download.py

# 각 노드에 필요한 파일 전송
for((i=2; i<=$NNODES; i++)); do
    scp ./scaling_factor_validation.py ./download.py worker$i:~/ &
done
wait

# 데이터 셋 전송
for((i=2;i<=$NNODES;i++)); do
  ssh worker$i "mkdir -p ~/data"
  scp ./data/cifar-10-python.tar.gz worker$i:~/data/ &
done
wait

# 데이터 셋 압축해제
for((i=2;i<=$NNODES;i++)); do
  ssh worker$i "python3 download.py" &
done
wait
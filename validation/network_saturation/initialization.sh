#!/bin/bash
TRAIN_NODES=17
REMOTE_NODES=1

python3 ./download.py

for((i=2; i<=$TRAIN_NODES; i++)); do
    scp ./DeepCheck_BASE.py ./network_saturation_validation.py ./compile.sh mpi_module.cpp ./download.py worker$i:~/ &
done
for((i=1; i<=$REMOTE_NODES; i++)); do
    scp ./DeepCheck_BASE.py ./network_saturation_validation.py ./compile.sh mpi_module.cpp remote$i:~/ &
done
wait

for((i=2;i<=$TRAIN_NODES;i++)); do
  ssh worker$i "mkdir -p ~/data"
  scp ./data/cifar-10-python.tar.gz worker$i:~/data/ &
done
wait

for((i=2;i<=$TRAIN_NODES;i++)); do
  ssh worker$i "python3 download.py" &
done
wait

for((i=2; i<=$TRAIN_NODES; i++)); do
  ssh worker$i "~/compile.sh" &
done
for((i=1; i<=$REMOTE_NODES; i++)); do
  ssh remote$i "~/compile.sh" &
done
wait
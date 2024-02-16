# 사용방법

> 5개의 파일과 temp 디렉토리가 준비되어있어야합니다.
> download.py, initialization.sh, run_experiments.sh, template, torch_save.py, temp (directory)

0. sudo vi /etc/hosts 를 통해 hosts파일에 다음과 같은 내용을 추가합니다. (미리 메모장 같은곳에 적어두었다가 붙여넣으시면 편합니다.)
```
172.31.0.1 worker1 
172.31.0.2 worker2
172.31.0.3 worker3
172.31.0.4 worker4
```
ip address는 각 노드들의 private ip address를 입력해야하시면 됩니다.
이때 worker1은 반드시 master가 되어야합니다.
1. 모든 sh파일에 실행권한을 추가해줍니다. (chmod +x *.sh)
2. initialization.sh을 실행합니다. 그러면,
   - master에서 데이터 셋 다운로드 및 압축 해제 (python3 ./download.py)
   - slave로 데이터셋과 학습코드 전송 (scp)
   - slave에서 원격으로 데이터셋 압축 해제를 수행합니다. (각 노드에서 python3 ./download.py를 수행)
3. run_experiments.sh의 매개변수들을 수정해주세요. (NNODES, MASTER_PORT, EPOCH). 이때, 에폭 수는 1로 수정합니다.
4. run_experiments.sh를 실행하여 학습을 1 에폭 수행합니다. (한 번 돌려야 그 이후 실험결과가 제대로 나옵니다.)
5. 학습이 잘 되었다면 원하는 에폭 수로 EPOCH값을 수정한 후 실험을 수행하시면됩니다.
6. torch_save.py에서 모델(resnet50)을 다른 모델로 수정하실 수도 있습니다. 수정하실 경우, 수정된 torch_save.py를 slave에게 다시 전송해주어야합니다.
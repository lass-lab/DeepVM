import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

# 데이터셋이 저장될 디렉토리 지정
data_root = './data'

# 변환 (transform)을 지정하면 데이터셋 로딩 시 적용됩니다. 
# 여기서는 Tensor로 변환하는 과정만 추가했으나, 필요한 경우 다른 전처리나 증강을 추가할 수 있습니다.
transform = transforms.Compose([
    transforms.ToTensor(),
])

# CIFAR10 데이터셋 다운로드
# train=True는 훈련 데이터를 다운로드하라는 의미입니다.
# download=True는 데이터셋이 지정된 경로에 없을 경우 다운로드하라는 의미입니다.
cifar10_train = CIFAR10(root=data_root, train=True, transform=transform, download=True)

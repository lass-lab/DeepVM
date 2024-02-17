import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet152, efficientnet_v2_l

from torch.utils.data import Dataset, DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from datetime import datetime
import time

def ddp_setup ():
    # print ("RANK: ", int (os.environ["RANK"]))
    init_process_group (backend="nccl")
    torch.cuda.set_device (int (os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__ (
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer
    ) -> None:
        self.local_rank = int (os.environ["LOCAL_RANK"])     # gpu_id 대신: 한 머신 안에서.
        self.global_rank = int (os.environ["RANK"])          # gpu_id 대신: 여러 머신에 걸쳐서.
        
        self.model = model.to (self.local_rank)     # local_rank의 디바이스로 전송.
        self.train_data = train_data
        self.optimizer = optimizer
        self.epochs_run = 0                     # CKP를 위함.

        self.model = DDP (self.model, device_ids=[self.local_rank])     # [DDP] init 타임에 DDP class로 wrapping.
        self.compute_duration = 0.0
        self.network_duration = 0.0

    def _run_epoch (self, epoch):
        # print (f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch (epoch)
        for source, targets in self.train_data:
            
            start_time = time.time()
            source = source.to (self.local_rank)
            targets = targets.to (self.local_rank)
            self.optimizer.zero_grad ()
            output = self.model (source)
            loss = F.cross_entropy (output, targets)
            loss.backward()
            self.compute_duration += time.time()-start_time

            start_time = time.time()
            torch.distributed.barrier()
            self.optimizer.step () # 그래디언트 업데이트
            self.network_duration += time.time()-start_time

    def train (self, max_epochs: int, model_name:str, result_file_name: str):
        for epoch in range (self.epochs_run, max_epochs):
            self._run_epoch (epoch)

        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if int(os.environ['RANK']) == 0:
            with open(result_file_name, 'a') as f:
                # if int(os.environ['WORLD_SIZE']) > 1:
                #     f.write(f"[{current_time}] training_time_{model_name}_{os.environ['WORLD_SIZE']}_nodes : {str(self.compute_duration)}, {str(self.network_duration)}\n")
                # else:
                f.write(f"[{current_time}] training_time_{model_name}_{os.environ['WORLD_SIZE']}_nodes : {str(self.compute_duration+self.network_duration)}\n")
                    

def get_model(model_name):
    if model_name == "resnet18":
        return resnet18(weights=None)
    elif model_name == "resnet34":
        return resnet34(weights=None)
    elif model_name == "resnet50":
        return resnet50(weights=None)
    elif model_name == "resnet152":
        return resnet152(weights=None)
    elif model_name == "effnetv2l":
        return efficientnet_v2_l(weights=None)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
def load_train_objs (model_name: str):
    #train_set = MyTrainDataset(2048)  # load your dataset
    transform = transforms.Compose ([
        transforms.ToTensor (),
        transforms.Normalize ((0.1307), (0.3081))
    ])    
    train_set = datasets.CIFAR10 ('data', train=True, download=True, transform=transform)
    model = get_model(model_name)
    optimizer = torch.optim.SGD (model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader (dataset: Dataset, batch_size: int):
    return DataLoader (
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler (dataset)
    )

def main (total_epochs: int, model_name:str, batch_size: int, result_file_name: str):
    ddp_setup ()
    dataset, model, optimizer = load_train_objs (model_name)
    train_data = prepare_dataloader (dataset, batch_size)
    trainer = Trainer (model, train_data, optimizer)
    trainer.train (total_epochs, model_name, result_file_name)
    destroy_process_group ()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser (description='simple distributed training job')
    parser.add_argument ('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument ('--model', type=str, default='resnet18', help='Model name')
    parser.add_argument ('--experiment_num', type=int, required=True, help='Experiment number to identify the output file')
    parser.add_argument ('--batch_size', default=128, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args ()
    
    world_size = torch.cuda.device_count ()
    # print ("world_size:", world_size)
    result_file_name = f"result_{args.experiment_num}.txt"
    main (args.total_epochs, args.model, args.batch_size, result_file_name)

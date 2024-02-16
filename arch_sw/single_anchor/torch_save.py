import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

import time

def ddp_setup ():
    print ("LOCAL_RANK: ", int (os.environ["LOCAL_RANK"]))
    init_process_group (backend="nccl")
    torch.cuda.set_device (int (os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__ (
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        save_every: int,
        snapshot_path: str,     # CKP 정보.
    ) -> None:
        self.local_rank = int (os.environ["LOCAL_RANK"])     # gpu_id 대신: 한 머신 안에서.
        self.global_rank = int (os.environ["RANK"])          # gpu_id 대신: 여러 머신에 걸쳐서.
        
        self.model = model.to (self.local_rank)     # local_rank의 디바이스로 전송.
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.epochs_run = 1                     # CKP를 위함.
        self.snapshot_path = snapshot_path

        self.model = DDP (self.model, device_ids=[self.local_rank])     # [DDP] init 타임에 DDP class로 wrapping.

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.epochs_run = snapshot['epoch']+1
        self.model.module.load_state_dict(snapshot['model_state_dict'])
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.scheduler.load_state_dict(snapshot['scheduler_state_dict'])
        
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch (self, source, targets):
        self.optimizer.zero_grad ()
        output = self.model (source)
        loss = F.cross_entropy (output, targets)
        loss.backward ()
        self.optimizer.step ()
        return loss.item()

    def _run_batch_val (self, source, targets):
        self.optimizer.zero_grad ()
        output = self.model (source)
        loss = F.cross_entropy (output, targets)
        return loss.item(), output

    def _run_epoch (self, epoch):
        start_time = time.time()

        #train
        self.model.train()
        b_sz = len (next (iter (self.train_data))[0])
        train_loss = 0
        size = len(self.train_data)
        print_period = min(100, int(size/3))
        print (f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch (epoch)
        for batch, (source, targets) in enumerate(self.train_data):
            source = source.to (self.local_rank)        # local_rank 디바이스로 전송.
            targets = targets.to (self.local_rank)      # local_rank 디바이스로 전송.
            loss = self._run_batch (source, targets)
            train_loss += loss
            if batch % print_period == 0:
                print(f"train loss: {loss:>7f} [{batch*len(source):>5d}/{size*b_sz:5d}]")

        train_loss = train_loss / size
        train_end_time = time.time()

        #validation
        self.model.eval()
        b_sz = len (next (iter (self.validation_data))[0])
        val_loss = 0
        correct = 0
        size = len(self.validation_data)
        with torch.no_grad():
            for batch, (source, targets) in enumerate(self.validation_data):
                source = source.to (self.local_rank)        # local_rank 디바이스로 전송.
                targets = targets.to (self.local_rank)      # local_rank 디바이스로 전송.
                loss, output = self._run_batch_val (source, targets)
                val_loss += loss
                prediction = output.max (1, keepdim = True) [1]
                correct += prediction.eq (targets.view_as (prediction)).sum ().item ()

        val_loss = val_loss / size
        val_accuracy = 100. * correct / len (self.validation_data.dataset)
        self.scheduler.step()
        val_end_time = time.time()

        print(f"[GPU{os.environ['RANK']}] Epoch {epoch} | n_duration = {val_end_time - start_time} | t_duration = {train_end_time - start_time} | v_duration = {val_end_time - train_end_time} | avg_train_loss = {train_loss} | avg_validation_loss = {val_loss} | validation_accuracy = {val_accuracy}")


    def _save_snapshot (self, epoch):
        # start_time = time.time()
        snapshot = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict (),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save (snapshot, self.snapshot_path)
        fd = os.open(self.snapshot_path, os.O_RDWR)
        os.fsync(fd)
        os.close(fd)
        # print(f"[{self.global_rank}] [{epoch}] [write] [{time.time()-start_time}]")
        print (f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train (self, max_epochs: int):
        for epoch in range (self.epochs_run, max_epochs+1):
            self._run_epoch (epoch)
            if epoch % self.save_every == 0 and self.global_rank == 0:
                self._save_snapshot (epoch)

def load_train_objs ():
    #train_set = MyTrainDataset(2048)  # load your dataset
    transform = transforms.Compose ([
        transforms.ToTensor (),
        transforms.Normalize ((0.1307), (0.3081))
    ])    
    train_set = datasets.CIFAR10 ('~/data', train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10 ('~/data', train=False, download=True, transform=transform)
    training_model = models.resnet50(num_classes=10)
    optimizer = torch.optim.Adam(training_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    return train_set, val_set, training_model, optimizer, scheduler


def prepare_dataloader (dataset: Dataset, batch_size: int, DDP:bool=True):
    if DDP:
        return DataLoader (
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler (dataset)
        )
    else:
        return DataLoader (
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
        )


def main (save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    main_start = time.time ()
    
    ddp_setup ()
    train_dataset, validation_dataset, model, optimizer, scheduler = load_train_objs ()
    train_data = prepare_dataloader (train_dataset, batch_size)
    valldation_data = prepare_dataloader(validation_dataset, batch_size, False)
    trainer = Trainer (model, train_data, valldation_data, optimizer, scheduler, save_every, snapshot_path)
    trainer.train (total_epochs)
    destroy_process_group ()
    
    if int (os.environ["RANK"]) == 0:
        main_end = time.time ()
        main_duration = main_end - main_start
        print(f"[{int(os.environ['RANK'])}] [makespan] [{main_duration}]")
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser (description='simple distributed training job')
    parser.add_argument ('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument ('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument ('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument ('--model_name', default='user_model', type=str, help='Input your ML model name (default: user_model)')
    args = parser.parse_args ()
    
    if int (os.environ["LOCAL_RANK"]) == 0:
        print(f"total_epochs: {args.total_epochs}")
        print(f"save_every: {args.save_every}")
        print(f"batch_size: {args.batch_size}")
        print(f"model_name: {args.model_name}")
    
    main (args.save_every, args.total_epochs, args.batch_size)

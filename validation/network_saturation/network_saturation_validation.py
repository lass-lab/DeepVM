import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import time

import DeepCheck_BASE

class Trainer:
    def __init__ (
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        validation_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        save_period: int,
        snapshot_path: str,
        train_node: DeepCheck_BASE.TRAIN,
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        
        self.model = model.to (self.local_rank)
        self.train_data = train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_period = save_period
        self.epochs_run = 1
        self.snapshot_path = snapshot_path
        self.train_node = train_node
        self.model = DDP (self.model, device_ids=[self.local_rank])
        if self.snapshot_path:
            assert os.path.exists (snapshot_path), f"can't open file \'{snapshot_path}\': No such file or directory"
            print ("Loading snapshot")
            self._load_snapshot (snapshot_path)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)

        self.epochs_run = snapshot['epoch']+1
        self.model.module.load_state_dict(snapshot['model_state_dict'])
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.scheduler.load_state_dict(snapshot['scheduler_state_dict'])
        
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _make_snapshot (self, epoch) -> dict:
        snapshot = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict (),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        return snapshot

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
        # 1. Train
        self.model.train()
        b_sz = len (next (iter (self.train_data))[0])
        train_loss = 0
        size = len(self.train_data)
        print_period = min(100, int(size/3))
        print (f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch (epoch)
        for batch, (source, targets) in enumerate(self.train_data):
            source = source.to (self.local_rank)
            targets = targets.to (self.local_rank)
            loss = self._run_batch (source, targets)
            train_loss += loss
            if batch % print_period == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | train loss: {loss:>7f} [{batch*len(source):>5d}/{size*b_sz:5d}]")

        train_loss = train_loss / size

        # 2. Validation
        self.model.eval()
        b_sz = len (next (iter (self.validation_data))[0])
        val_loss = 0
        correct = 0
        size = len(self.validation_data)
        with torch.no_grad():
            for batch, (source, targets) in enumerate(self.validation_data):
                source = source.to (self.local_rank)
                targets = targets.to (self.local_rank)
                loss, output = self._run_batch_val (source, targets)
                val_loss += loss
                prediction = output.max (1, keepdim = True) [1]
                correct += prediction.eq (targets.view_as (prediction)).sum ().item ()

        val_loss = val_loss / size
        self.scheduler.step(val_loss)

        val_accuracy = 100. * correct / len (self.validation_data.dataset)
        print(f"[GPU{self.global_rank}] Epoch {epoch} | avg_train_loss = {train_loss} | avg_validation_loss = {val_loss} | validation_accuracy = {val_accuracy}")

    def train(self, max_epochs: int):
        for epoch in range (self.epochs_run, max_epochs+1):
            self._run_epoch (epoch)
            if epoch % self.save_period == 0:
                self.train_node.save(self._make_snapshot(epoch)) # [DeepCheck] Save model data

def load_train_objs ():
    transform = transforms.Compose ([
        transforms.ToTensor (),
        transforms.Normalize ((0.1307), (0.3081))
    ])    
    train_set = datasets.CIFAR10 ('data', train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10 ('data', train=False, download=True, transform=transform)

    '''
    If you want to use imageNet:
        # train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor (),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        # val_transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor (),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        # train_set = datasets.ImageNet('/mnt/efs/imagenet', split='train', transform=train_transform)
        # val_set = datasets.ImageNet('/mnt/efs/imagenet', split='val', transform=val_transform)
    '''

    model = models.resnet50(num_classes=10, pretrained=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=False)

    return train_set, val_set, model, optimizer, scheduler

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

def main (save_period: int, total_epochs: int, batch_size: int, snapshot_path: str, train_node: DeepCheck_BASE.TRAIN):
    train_dataset, validation_dataset, model, optimizer, scheduler = load_train_objs ()
    train_data = prepare_dataloader (train_dataset, batch_size)
    valldation_data = prepare_dataloader(validation_dataset, batch_size, False)
    trainer = Trainer (model, train_data, valldation_data, optimizer, scheduler, save_period, snapshot_path, train_node)
    start_time = time.time()
    trainer.train (total_epochs)
    if int(os.environ['RANK']) == 0:
        print(f'makespan: {time.time()-start_time}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser (description='simple distributed training job')
    parser.add_argument ('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument ('save_period', type=int, help='How often to save a snapshot')
    parser.add_argument ('train_count', type=int, help='The number of training nodes')
    parser.add_argument ('remote_count', type=int, help='The number of remote nodes')
    parser.add_argument ('training_master_addr', type=str, help='Training node IP address with rank 0')
    parser.add_argument ('training_master_port', type=str, help="Training node port number with rank 0")
    parser.add_argument ('--starting_epoch', default=1, type=int, help='Input the start epoch for the DeepCheck module. If epoch information exists in the snapshot file, that information will take precedence. (default: 1)')
    parser.add_argument ('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument ('--shard_size', default=1, type=int, help='Input size of sharding (default: 1)')
    parser.add_argument ('--remote_buffer_size', default=1, type=int, help='Input data buffer size of remote node. (default: 1)')
    parser.add_argument ('--model_name', default='user_model', type=str, help='Input your ML model name (default: user_model)')
    parser.add_argument ('--file_name_include_datetime', '--file_date', default=False, type=bool, help='Add datetime to checkpoint files. (default: False)')
    parser.add_argument ('--file_save_in_dictionary', '--file_dictionary', default=False, type=bool, help='Checkfiles will be saved in a dictionary (default: False)')
    parser.add_argument ('--snapshot_path', default=None, type=str, help='Input checkpoint file name (default: None)')
    args = parser.parse_args ()

    communicator, train_node, _ = DeepCheck_BASE.init_DeepCheck(args) # [DeepCheck] Initialize the DeepCheck environment.

    main(args.save_period, args.total_epochs, args.batch_size, args.snapshot_path, train_node)

    DeepCheck_BASE.destroy_DeepCheck() # [DeepCheck] Destroy the DeepCheck environment.

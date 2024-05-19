import os

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torchvision.datasets import CIFAR10

from models import env
from models.cifar10.models import ResNet18
from models.trainer import Trainer
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel

from models.tool import Statistics


class Cifar10Trainer(Trainer):
    def __init__(self, args):
        super().__init__()
        # Args
        self.args = args
        self.lr = 0.008
        self.start_epoch = 0
        self.data_dir = env.get_dataset_path("cifar10")
        self.checkpoint_path = env.get_checkpoint_path(args.job_name)
        self.device = torch.device(f"cuda:{self.args.device}")

        # Model
        print('==> Building model..')
        self.model = ResNet18().to(self.args.device)
        self.model = DistributedDataParallel(self.model, device_ids=[self.args.device], output_device=self.args.device)
        self.criterion = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = ExponentialLR(self.optimizer, 0.0133 ** (1.0 / self.args.max_epoch))

        # recover for elastic training
        self.load_checkpoint()

        # Data
        print('==> Preparing data..')
        train_set = CIFAR10(root=self.data_dir, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]))
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_sampler.set_epoch(self.start_epoch)
        train_loader = DataLoader(train_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=train_sampler,
                                  pin_memory=True)

        val_set = CIFAR10(root=self.data_dir, train=False, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                          ]))
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=True)
        val_sampler.set_epoch(self.start_epoch)
        val_loader = DataLoader(val_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=val_sampler,
                                pin_memory=True)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # metrics
        self.train_metric = Statistics(["loss", "accuracy"], device=self.args.device,
                                       acc_steps=self.args.acc_step)
        self.val_metric = Statistics(["loss", "accuracy"], device=self.args.device,
                                     acc_steps=self.args.acc_step)

    def train_acc_step(self, i, batch):
        inputs, targets = [item.to(self.args.device) for item in batch]
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets) / self.args.acc_step
        loss.backward()

        loss = loss.item()
        accuracy = outputs.max(1)[1].eq(targets).sum().item() / targets.size(0) / self.args.acc_step
        self.train_metric.accumulate_in_batch([loss, accuracy])

    def val_acc_step(self, i, batch):
        inputs, targets = [item.to(self.args.device) for item in batch]
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets) / self.args.acc_step

        loss = loss.item()
        accuracy = outputs.max(1)[1].eq(targets).sum().item() / targets.size(0) / self.args.acc_step
        self.val_metric.accumulate_in_batch([loss, accuracy])

    def save_checkpoint(self, epoch):
        if dist.get_rank() == 0:
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch
            }, self.checkpoint_path)
            print(f"save checkpoint to {self.checkpoint_path}")

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print(f"==> no checkpoint found at '{self.checkpoint_path}')")
            return
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"]
        print(f"load checkpoint from {self.checkpoint_path}")

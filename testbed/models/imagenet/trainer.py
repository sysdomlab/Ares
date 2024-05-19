import os

import torch
import torchvision
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from torchvision.datasets import ImageFolder

from models import env
from models.trainer import Trainer
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel

from models.tool import Statistics


def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * 100.0 / batch_size, correct_k.item(), batch_size))
        return res


class ImagenetTrainer(Trainer):
    def __init__(self, args):
        super().__init__()
        # Args
        self.args = args
        self.lr = 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.start_epoch = 0
        self.data_dir = env.get_dataset_path("imagenet")
        self.checkpoint_path = env.get_checkpoint_path(args.job_name)
        self.device = torch.device(f"cuda:{self.args.device}")

        # Model
        print('==> Building model..')
        self.model = torchvision.models.resnet50().to(self.args.device)
        self.model = DistributedDataParallel(self.model, device_ids=[self.args.device], output_device=self.args.device)
        self.criterion = CrossEntropyLoss().to(self.args.device)
        self.optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                             weight_decay=self.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, 0.0133 ** (1.0 / self.args.max_epoch))

        # recover for elastic training
        self.load_checkpoint()

        # Data
        print('==> Preparing data..')
        train_set = ImageFolder(root=self.data_dir + "/train",
                                transform=transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ]))
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_sampler.set_epoch(self.start_epoch)
        train_loader = DataLoader(train_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=train_sampler,
                                  pin_memory=True)

        val_set = ImageFolder(root=self.data_dir + "/val",
                              transform=transforms.Compose([
                                  transforms.Resize(256),
                                  transforms.CenterCrop(224),
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
        self.train_metric = Statistics(["loss", "accuracy1", "accuracy5"], device=self.args.device,
                                       acc_steps=self.args.acc_step)
        self.val_metric = Statistics(["loss", "accuracy1", "accuracy5"], device=self.args.device,
                                     acc_steps=self.args.acc_step)

    def train_acc_step(self, i, batch):
        inputs, targets = [item.to(self.args.device) for item in batch]
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets) / self.args.acc_step
        loss.backward()

        (acc1, correct1, total1), (acc5, correct5, total5) = compute_accuracy(outputs, targets, topk=(1, 5))
        loss = loss.item()
        accuracy1 = acc1[0] / self.args.acc_step
        accuracy5 = acc5[0] / self.args.acc_step
        self.train_metric.accumulate_in_batch([loss, accuracy1, accuracy5])

    def val_acc_step(self, i, batch):
        inputs, targets = [item.to(self.args.device) for item in batch]
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets) / self.args.acc_step

        (acc1, correct1, total1), (acc5, correct5, total5) = compute_accuracy(outputs, targets, topk=(1, 5))
        loss = loss.item()
        # print(acc1, correct1, total1, acc5, correct5, total5)
        accuracy1 = acc1[0] / self.args.acc_step
        accuracy5 = acc5[0] / self.args.acc_step
        self.val_metric.accumulate_in_batch([loss, accuracy1, accuracy5])

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

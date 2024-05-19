import os

import torch
import torch.distributed as dist
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DistributedSampler, DataLoader

from models import env
from models.ncf import data_utils, config
from models.ncf.evaluate import hit, ndcg
from models.ncf.model import NCF
from models.trainer import Trainer
from torch.nn.parallel import DistributedDataParallel

from models.tool import Statistics


class NCFTrainer(Trainer):
    def __init__(self, args):
        super().__init__()
        # Args
        self.args = args
        self.lr = 0.00001
        self.factor_num = 32
        self.num_layers = 3
        self.dropout = 0.0
        self.num_ng = 4
        self.top_k = 10
        self.start_epoch = 0
        self.data_dir = env.get_dataset_path("ncf")
        self.checkpoint_path = env.get_checkpoint_path(args.job_name)
        self.device = torch.device(f"cuda:{self.args.device}")

        # Model
        print('==> Building model..')
        train_data, test_data, user_num, item_num, train_mat = data_utils.load_all(data_path=self.data_dir)
        self.model = NCF(user_num, item_num, self.factor_num, self.num_layers,
                         self.dropout, config.model).to(self.args.device)
        self.model = DistributedDataParallel(self.model, device_ids=[self.args.device], output_device=self.args.device)
        self.criterion = BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), betas=(0.9, 0.999), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [4, 7, 10], gamma=0.2)

        # recover for elastic training
        self.load_checkpoint()

        # Data
        print('==> Preparing data..')
        train_set = data_utils.NCFData(train_data, item_num, train_mat, self.num_ng, True)
        train_set.ng_sample()
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_sampler.set_epoch(self.start_epoch)
        train_loader = DataLoader(train_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=train_sampler,
                                  pin_memory=True)

        val_set = data_utils.NCFData(test_data, item_num, train_mat, 0, False)
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=True)
        val_sampler.set_epoch(self.start_epoch)
        val_loader = DataLoader(val_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=val_sampler,
                                pin_memory=True)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # metrics
        self.train_metric = Statistics(["loss", "HR", "NDCG"], device=self.args.device,
                                       acc_steps=self.args.acc_step)
        self.val_metric = Statistics(["loss", "HR", "NDCG"], device=self.args.device,
                                     acc_steps=self.args.acc_step)

    def train_acc_step(self, i, batch):
        user, item, label = [item.to(self.args.device) for item in batch]
        label = label.float()
        prediction = self.model(user, item)
        loss = self.criterion(prediction, label) / self.args.acc_step
        loss.backward()

        _, indices = torch.topk(prediction, self.top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        gt_item = item[0].item()
        HR = hit(gt_item, recommends) / self.args.acc_step
        NDCG = ndcg(gt_item, recommends) / self.args.acc_step
        self.train_metric.accumulate_in_batch([loss.item(), HR, NDCG])

    def val_acc_step(self, i, batch):
        user, item, label = [item.to(self.args.device) for item in batch]
        label = label.float()
        prediction = self.model(user, item)
        loss = self.criterion(prediction, label) / self.args.acc_step

        _, indices = torch.topk(prediction, self.top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()
        gt_item = item[0].item()
        HR = hit(gt_item, recommends) / self.args.acc_step
        NDCG = ndcg(gt_item, recommends) / self.args.acc_step
        self.train_metric.accumulate_in_batch([loss.item(), HR, NDCG])

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

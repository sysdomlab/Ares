import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DistributedSampler, DataLoader

from models import env
from models.ares_trainer import AresTrainer
# from torch.nn.parallel import DistributedDataParallel as DDP
from models.tool import AresDataParallel as DDP

from models.yolov3.model.loss.yolo_loss import YoloV3Loss
from models.yolov3.model.yolov3 import Yolov3
from models.yolov3.utils import VocDataset
from models.tool import Statistics
import config.yolov3_config_voc as cfg


class YOLOv3AresTrainer(AresTrainer):
    def __init__(self, args):
        super().__init__()
        # Args
        self.args = args
        self.lr = 0.008
        self.start_epoch = 0
        self.data_dir = env.get_dataset_path("yolov3")
        self.checkpoint_path = env.get_checkpoint_path(args.job_name)
        self.device = torch.device(f"cuda:{self.args.device}")

        # Model
        print('==> Building model..')
        self.model = Yolov3().to(self.args.device)
        # self.model.load_darknet_weights(weight_path)
        self.model = DDP(self.model, device_ids=[self.args.device], output_device=self.args.device)
        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])
        self.optimizer = SGD(self.model.parameters(), lr=cfg.TRAIN["LR_INIT"],
                             momentum=cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
        self.scheduler = CosineAnnealingLR(self.optimizer,
                                           T_max=self.args.max_epoch,
                                           eta_min=cfg.TRAIN["LR_END"])

        # recover from checkpoint
        self.load_checkpoint()

        # Data
        print('==> Preparing data..')
        train_set = VocDataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"], data_path=self.data_dir)
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_sampler.set_epoch(self.start_epoch)
        train_loader = DataLoader(train_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=train_sampler,
                                  pin_memory=True)

        val_set = VocDataset(anno_file_type="test", data_path=self.data_dir)
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=True)
        val_sampler.set_epoch(self.start_epoch)
        val_loader = DataLoader(val_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=val_sampler,
                                pin_memory=True)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # metrics
        self.train_metric = Statistics(["loss", "loss_giou", "loss_conf", "loss_cls"], device=self.args.device,
                                       acc_steps=self.args.acc_step)
        self.val_metric = Statistics(["loss", "loss_giou", "loss_conf", "loss_cls"], device=self.args.device,
                                     acc_steps=self.args.acc_step)

    def train_acc_step(self, i, batch):
        imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = (i.to(self.args.device) for i in batch)
        p, p_d = self.model(imgs)
        loss, loss_giou, loss_conf, loss_cls \
            = self.criterion(p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
        loss = loss / self.args.acc_step
        loss_giou = loss_giou / self.args.acc_step
        loss_conf = loss_conf / self.args.acc_step
        loss_cls = loss_cls / self.args.acc_step
        loss.backward()

        self.train_metric.accumulate_in_batch([loss.item(), loss_giou.item(), loss_conf.item(), loss_cls.item()])

    def val_acc_step(self, i, batch):
        if not self.model.training:
            self.model.train()
        imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = (i.to(self.args.device) for i in batch)
        p, p_d = self.model(imgs)
        loss, loss_giou, loss_conf, loss_cls \
            = self.criterion(p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
        loss = loss / self.args.acc_step
        loss_giou = loss_giou / self.args.acc_step
        loss_conf = loss_conf / self.args.acc_step
        loss_cls = loss_cls / self.args.acc_step

        self.val_metric.accumulate_in_batch([loss.item(), loss_giou.item(), loss_conf.item(), loss_cls.item()])

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

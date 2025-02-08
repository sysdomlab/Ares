import json
import os

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler

from models import env
from models.deepspeech2.data.data_loader import SpectrogramDataset, AudioDataLoader
from models.deepspeech2.decoder import GreedyDecoder
from models.deepspeech2.model import supported_rnns, DeepSpeech
from models.ares_trainer import AresTrainer
# from torch.nn.parallel import DistributedDataParallel as DDP
from models.tool import AresDataParallel as DDP

from models.tool import Statistics


class Deepspeech2AresTrainer(AresTrainer):
    def __init__(self, args):
        super().__init__()
        # Args
        self.args = args
        self.start_epoch = 0
        self.data_dir = env.get_dataset_path("deepspeech2")
        self.checkpoint_path = env.get_checkpoint_path(args.job_name)
        self.device = torch.device(f"cuda:{self.args.device}")

        self.lr = 0.008
        self.labels_path = f"{self.data_dir}/labels.json"
        self.sample_rate = 16000
        self.window_size = .02
        self.window_stride = .01
        self.window = 'hamming'
        self.noise_dir = None
        self.noise_prob = 0.4
        self.noise_min = 0.0
        self.noise_max = 0.5
        self.rnn_type = 'gru'
        self.hidden_size = 800
        self.hidden_layers = 5
        self.bidirectional = False
        self.momentum = 0.9
        self.train_manifest = f'{self.data_dir}/train_manifest.csv'
        self.val_manifest = f'{self.data_dir}/val_manifest.csv'
        self.speed_volume_perturb = False
        self.max_norm = 1
        self.spec_augment = False

        with open(self.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=self.sample_rate,
                          window_size=self.window_size,
                          window_stride=self.window_stride,
                          window=self.window,
                          noise_dir=self.noise_dir,
                          noise_prob=self.noise_prob,
                          noise_levels=(self.noise_min, self.noise_max))

        # Model
        print('==> Building model..')
        self.model = DeepSpeech(rnn_hidden_size=self.hidden_size,
                                nb_layers=self.hidden_layers,
                                labels=labels,
                                rnn_type=supported_rnns[self.rnn_type.lower()],
                                audio_conf=audio_conf,
                                bidirectional=self.bidirectional).to(self.args.device)
        self.model = DDP(self.model, device_ids=[self.args.device], output_device=self.args.device)
        self.decoder = GreedyDecoder(labels)
        self.criterion = torch.nn.CTCLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, nesterov=False)
        self.scheduler = ExponentialLR(self.optimizer, 0.0133 ** (1.0 / self.args.max_epoch))

        # recover for elastic training
        self.load_checkpoint()

        # Data
        print('==> Preparing data..')
        # todo train_manifest 中使用的是 /datasets/voxforge，我们的路径是 /home/cchen/yfliu/ares/data/voxforge
        train_set = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=self.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=self.speed_volume_perturb,
                                       spec_augment=self.spec_augment,
                                       samples=4074
                                       )
        # according to the length of voxforge_cmu_us_american_train_manifest_part1.csv
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_sampler.set_epoch(self.start_epoch)
        train_loader = AudioDataLoader(train_set, self.args.acc_bsz, shuffle=False, num_workers=20,
                                       sampler=train_sampler, pin_memory=True)

        val_set = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=self.val_manifest, labels=labels,
                                     normalize=True, speed_volume_perturb=False, spec_augment=False,
                                     samples=452
                                     )
        # according to the length of voxforge_cmu_us_american_train_manifest_part2.csv
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=True)
        val_sampler.set_epoch(self.start_epoch)
        val_loader = AudioDataLoader(val_set, self.args.acc_bsz, shuffle=False, num_workers=20,
                                     sampler=val_sampler, pin_memory=True)

        self.train_loader = train_loader
        self.val_loader = val_loader

        # metrics  Word Error Rate, Character Error Rate
        self.train_metric = Statistics(["loss", "wer", "cer"], device=self.args.device,
                                       acc_steps=self.args.acc_step)
        self.val_metric = Statistics(["loss", "wer", "cer"], device=self.args.device,
                                     acc_steps=self.args.acc_step)

    def validate(self, out, targets, output_sizes, target_sizes):
        # unflatten targets
        split_targets, offset = [], 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size
        decoded_output, _ = self.decoder.decode(out, output_sizes)
        target_strings = self.decoder.convert_to_strings(split_targets)
        wer_total, cer_total, num_tokens, num_chars = 0, 0, 0, 0
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer_total += self.decoder.wer(transcript, reference)
            cer_total += self.decoder.cer(transcript, reference)
            num_tokens += len(reference.split())
            num_chars += len(reference.replace(' ', ''))
        wer = wer_total / num_tokens * 100
        cer = cer_total / num_chars * 100
        return wer, cer

    def train_acc_step(self, i, batch):
        inputs, targets, input_sizes, target_sizes = batch
        inputs = inputs.to(self.args.device)
        input_sizes = input_sizes.to(self.args.device)
        out, output_sizes = self.model(inputs, input_sizes)
        float_out = out.transpose(0, 1).float()
        loss = self.criterion(float_out, targets, output_sizes, target_sizes) / self.args.acc_step
        loss.backward()

        wer, cer = self.validate(out, targets, output_sizes, target_sizes)
        self.train_metric.accumulate_in_batch([loss.item(), wer / self.args.acc_step, cer / self.args.acc_step])

    def val_acc_step(self, i, batch):
        inputs, targets, input_sizes, target_sizes = batch
        inputs = inputs.to(self.args.device)
        input_sizes = input_sizes.to(self.args.device)
        out, output_sizes = self.model(inputs, input_sizes)
        float_out = out.transpose(0, 1).float()
        loss = self.criterion(float_out, targets, output_sizes, target_sizes) / self.args.acc_step

        wer, cer = self.validate(out, targets, output_sizes, target_sizes)
        self.val_metric.accumulate_in_batch([loss.item(), wer / self.args.acc_step, cer / self.args.acc_step])

    def save_checkpoint(self, epoch):
        if dist.get_rank() == 0:
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch
            }, self.checkpoint_path)
            print(f"==> save checkpoint to {self.checkpoint_path}")

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

import os

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DistributedSampler, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadResult

from models import env
from models.trainer import Trainer
from torch.nn.parallel import DistributedDataParallel

from models.tool import Statistics


def load_cache(app_args, evaluate=False):
    # Load data features from cache or dataset file
    input_dir = app_args.data_dir if app_args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, app_args.model_name_or_path.split("/"))).pop(),
            str(app_args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    assert os.path.exists(cached_features_file), "No cached features found at {}".format(cached_features_file)
    print("==> Loading features from cached file %s", cached_features_file)
    features_and_dataset = torch.load(cached_features_file)

    return features_and_dataset["dataset"], features_and_dataset["examples"], features_and_dataset["features"]


class BertTrainer(Trainer):
    def __init__(self, args):
        # require transformers==3.4.0 protobuf==3.20.0
        super().__init__()
        # Args
        self.args = args
        self.data_dir = env.get_dataset_path("bert")
        self.checkpoint_path = env.get_checkpoint_path(args.job_name)
        self.start_epoch = 0
        self.device = torch.device(f"cuda:{self.args.device}")

        self.model_type = "bert"
        self.model_name_or_path = f"{self.data_dir}/model/bert-base-uncased"
        self.output_dir = env.get_checkpoint_path(args.job_name, return_dir=True)  # todo
        self.data_dir = f"{self.data_dir}/data"
        self.version_2_with_negative = False
        self.null_score_diff_threshold = 0.0
        self.max_seq_length = 384
        self.do_lower_case = True
        self.per_gpu_train_batch_size = 12
        self.per_gpu_eval_batch_size = 8
        self.learning_rate = 3e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 2.0
        self.warmup_steps = 0
        self.n_best_size = 20
        self.max_answer_length = 30
        self.verbose_logging = True

        # Model
        print('==> Building model..')
        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, do_lower_case=self.do_lower_case)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name_or_path, config=self.config)
        self.model = self.model.to(self.device)
        self.model = DistributedDataParallel(self.model, device_ids=[self.args.device], output_device=self.args.device)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay, },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.scheduler = ExponentialLR(self.optimizer, 0.0133 ** (1.0 / self.args.max_epoch))

        # recover for elastic training
        self.load_checkpoint()

        # Data
        print('==> Preparing data..')
        train_set, _, _ = load_cache(self, evaluate=False)
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_sampler.set_epoch(self.start_epoch)
        train_loader = DataLoader(train_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=train_sampler,
                                  pin_memory=True)

        val_set, examples, features = load_cache(self, evaluate=True)
        val_sampler = DistributedSampler(val_set, shuffle=False, drop_last=True)
        val_sampler.set_epoch(self.start_epoch)
        val_loader = DataLoader(val_set, self.args.acc_bsz, shuffle=False, num_workers=20, sampler=val_sampler,
                                pin_memory=True)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.examples = examples
        self.features = features

        # metrics
        self.train_metric = Statistics(["loss"], device=self.args.device,
                                       acc_steps=self.args.acc_step)
        self.val_metric = Statistics(["f1"], device=self.args.device,
                                     acc_steps=self.args.acc_step)
        self.all_results = []

    def train_acc_step(self, i, batch):
        batch = tuple(t.to(self.device) for t in batch)
        inputs = {"input_ids": batch[0],
                  "attention_mask": batch[1],
                  "start_positions": batch[3],
                  "end_positions": batch[4]}

        outputs = self.model(**inputs)
        loss = outputs[0] / self.args.acc_step

        loss.backward()

        # todo
        # if train_dataloader._elastic.is_sync_step():
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        self.train_metric.accumulate_in_batch([loss.item()])

    def val_acc_step(self, i, batch):
        batch = tuple(t.to(self.device) for t in batch)
        outputs = self.model(**{"input_ids": batch[0],
                                "attention_mask": batch[1],
                                "return_dict": False})

        for i, feature_index in enumerate(batch[3]):  # feature_indices = batch[3]
            output = [output[i].detach().cpu().tolist() for output in outputs]
            start_logits, end_logits = output
            unique_id = int(self.features[feature_index.item()].unique_id)
            result = SquadResult(unique_id, start_logits, end_logits)
            self.all_results.append(result)

        # at end of len(dataloader)
        if i == len(self.val_loader) - 1:
            self.val_metric.reset()
            # Compute predictions
            predictions = compute_predictions_logits(self.examples,
                                                     self.features,
                                                     self.all_results,
                                                     self.n_best_size,
                                                     self.max_answer_length,
                                                     self.do_lower_case,
                                                     None,
                                                     None,
                                                     None,
                                                     self.verbose_logging,
                                                     self.version_2_with_negative,
                                                     self.null_score_diff_threshold,
                                                     self.tokenizer)
            results = squad_evaluate(self.examples, predictions)  # Compute the F1 and exact scores.
            self.val_metric.accumulate_in_batch([results["f1"]])

    def save_checkpoint(self, epoch):
        if dist.get_rank() == 0:
            torch.save({
                "model": self.model.module.state_dict(),  # todo check other model
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
        self.model.module.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"]
        print(f"load checkpoint from {self.checkpoint_path}")

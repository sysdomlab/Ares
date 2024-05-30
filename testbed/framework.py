import argparse
import json
import os
import signal
import time

import torch
import torch.distributed as dist

from models import env
from models.tool import set_random_seed, signal_handler, get_signal_received, Statistics
from models.cifar10.trainer import Cifar10Trainer
from models.imagenet.trainer import ImagenetTrainer
from models.yolov3.trainer import YOLOv3Trainer
from models.ncf.trainer import NCFTrainer
from models.deepspeech2.trainer import Deepspeech2Trainer
from models.bert.trainer import BertTrainer


def get_trainer_cls(model_name):
    return {
        'cifar10': Cifar10Trainer,
        'imagenet': ImagenetTrainer,
        'yolov3': YOLOv3Trainer,
        'ncf': NCFTrainer,
        'deepspeech2': Deepspeech2Trainer,
        'bert': BertTrainer,
    }[model_name]


def load_job_state(job_name):
    job_state_path = env.get_job_state_path(job_name)
    if not os.path.exists(job_state_path):
        os.makedirs(os.path.dirname(job_state_path), exist_ok=True)
        return {
            'epoch': 0,
            'completed': False,
        }
    with open(job_state_path, 'r') as f:
        state = json.load(f)
    return state


def save_job_state(job_name, content: dict):
    if dist.get_rank() == 0:
        job_state_path = env.get_job_state_path(job_name)
        with open(job_state_path + ".tmp", 'w') as f:
            json.dump(content, f)
        os.rename(job_state_path + ".tmp", job_state_path)  # atomic write


def is_sync_step(iterations, acc_step):
    return (iterations + 1) % acc_step == 0


def is_last_step(iterations, loader):
    return iterations == len(loader) - 1


def main(args):
    start = time.time()
    timer_1 = time.time()
    dist.init_process_group(backend=args.backend,
                            init_method=f"tcp://{args.master_address}:{args.master_port}",
                            world_size=args.world_size,
                            rank=args.global_rank)

    print(args.__dict__)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    set_random_seed()
    # 1. initialize args, recover from checkpoint, prepare dataset
    trainer = get_trainer_cls(args.model_name)(args)

    print(f"*** start training ***")
    print(f"  job_name: {args.job_name}")
    print(f"  world_size = {args.world_size}, global_rank = {args.global_rank}, device = {args.device}")
    print(f"  epochs = {args.max_epoch}, iterations = {len(trainer.train_loader)}")
    print(f"  acc_bsz = {args.acc_bsz}, acc_step = {args.acc_step}")
    print(f"  local_bsz = {args.acc_bsz * args.acc_step}, "
          f"global_bsz = {args.acc_bsz * args.acc_step * args.world_size}")

    # 2. train loop
    performance_metric = Statistics(["batch", "sync", "gpu", "data", "other"],
                                    args.device, acc_steps=args.acc_step)
    # DLT jobs' training progress(e.g. epoch, iteration) is the only thing that Ares needs to track.
    # Other functions like gradient accumulation, model saving, graceful exit, etc. are application-specific.
    job_state = load_job_state(args.job_name)
    start_epoch = int(job_state["epoch"])
    start_iter = int((job_state["epoch"] - int(job_state["epoch"])) * len(trainer.train_loader))

    dist.barrier()
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 正常终止信号

    print(f">>> initializing time {time.time() - timer_1}s")
    print(f">>> start from epoch {start_epoch}, iter {start_iter}")
    for epoch in range(start_epoch, args.max_epoch):  # set start_epoch from checkpoint
        # 2.1 train for one epoch
        timer_1, timer_2, first_batch = time.time(), time.time(), True
        gpu_time, data_time, other_time = 0, 0, 0
        trainer.model.train()
        for i, batch in enumerate(trainer.train_loader):
            # 2.1.1 train for one acc_bsz
            if epoch == start_epoch and i < start_iter:  # set start_iter from checkpoint
                dist.barrier()
                if get_signal_received():
                    exit(143)
                continue

            torch.cuda.synchronize()
            data_time = time.time() - timer_2
            timer_2 = time.time()
            # 2.1.2 Multiple gradient accumulations constitute a complete gradient update
            if not is_sync_step(i, args.acc_step) and not is_last_step(i, trainer.train_loader):
                with trainer.model.no_sync():
                    trainer.train_acc_step(i, batch)
            else:
                trainer.train_acc_step(i, batch)  # gradient all-reduce occurs here
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

            torch.cuda.synchronize()
            gpu_time = time.time() - timer_2
            timer_2 = time.time()
            performance_metric.accumulate_in_batch([data_time + gpu_time + other_time,
                                                    trainer.model.get_sync_time(),
                                                    gpu_time, data_time, other_time])

            # 2.1.3 print training info
            if is_sync_step(i, args.acc_step) or is_last_step(i, trainer.train_loader):
                trainer.train_metric.update_global()
                if not is_last_step(i, trainer.train_loader):  # some app may use drop_last=False
                    performance_metric.update_global()
                # 2.1.1.1 print training info
                # ignore the first batch's performance metric
                if (((i + 1) % (args.acc_step * args.print_freq) == 0 or is_last_step(i, trainer.train_loader))
                        and not first_batch):
                    print(f'[Epoch {epoch}][{((i / len(trainer.train_loader)) * 100):.2f}%]:'
                          f'{performance_metric}{trainer.train_metric}')
                    if args.profile:
                        dist.barrier()
                        exit(0)
                if first_batch or is_last_step(i, trainer.train_loader):
                    performance_metric.reset()
                    first_batch = False

                # 2.1.1.3 save persistent state
                save_job_state(args.job_name, {
                    'epoch': epoch + (i + 1) / len(trainer.train_loader),
                    'completed': False
                })  # save Ares job state
                dist.barrier()
                if get_signal_received():  # graceful exit
                    trainer.save_checkpoint(epoch)  # save application-specific checkpoint
                    exit(143)

            other_time = time.time() - timer_2
            timer_2 = time.time()

        trainer.scheduler.step()
        trainer.save_checkpoint(epoch)

        # 2.2 validate for one epoch
        trainer.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(trainer.val_loader):
                trainer.val_acc_step(i, batch)

                if is_sync_step(i, args.acc_step) or is_last_step(i, trainer.val_loader):
                    trainer.val_metric.update_global()

                    dist.barrier()
                    if get_signal_received():
                        print("exit at validation")
                        exit(143)

        print(f'Finish Epoch {epoch}({time.time() - timer_1:.2f}s): val_metric {trainer.val_metric}')

    print(f"train complete in {time.time() - start}s")
    dist.barrier()
    save_job_state(args.job_name, {
        'epoch': args.max_epoch,
        'completed': True
    })
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, default='deepspeech2-11')
    parser.add_argument('--master_address', type=str, default='10.0.0.24')
    parser.add_argument('--master_port', type=str, default='17001')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--acc_bsz', type=int, default=80)
    parser.add_argument('--acc_step', type=int, default=1)

    parser.add_argument('--backend', type=str, default="nccl")

    parser.add_argument('--model_name', type=str, default='deepspeech2')
    parser.add_argument('--max_epoch', type=int, default=20)

    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--profile', action='store_true')

    main(parser.parse_args())

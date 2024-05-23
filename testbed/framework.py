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


def load_job_state(dataloader_len, job_name):
    job_state_path = env.get_job_state_path(job_name)
    if not os.path.exists(job_state_path):
        os.makedirs(os.path.dirname(job_state_path), exist_ok=True)
        return 0, 0, False
    with open(job_state_path, 'r') as f:
        state = json.load(f)
    epoch = state['epoch']
    completed = state['completed']
    return int(epoch), int((epoch - int(epoch)) * dataloader_len), completed


def save_job_state(job_name, epoch, iterations, dataloader_len, completed=False):
    if dist.get_rank() == 0:
        file_name = env.get_job_state_path(job_name)
        with open(file_name + ".tmp", 'w') as f:
            json.dump({
                'epoch': epoch + iterations / dataloader_len,
                'completed': completed
            }, f)
        os.rename(file_name + ".tmp", file_name)  # atomic write


def is_sync_step(iterations, acc_step, trainer):
    return (iterations + 1) % acc_step == 0 or iterations == len(trainer.val_loader) - 1


def main(args):
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
    performance_metric = Statistics(["batch", "gpu", "data", "sync"], args.device, acc_steps=args.acc_step)
    # DLT jobs' training progress(e.g. epoch, iteration) is the only thing that Ares needs to track.
    # Other functions like gradient accumulation, model saving, graceful exit, etc. are application-specific.
    start_epoch, start_iter, completed = load_job_state(len(trainer.train_loader), args.job_name)
    if completed:
        print(f"job {args.job_name} has already completed, exit")
        exit(0)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 正常终止信号
    print(f">>> initializing time {time.time() - timer_1}s")
    print(f">>> start from epoch {start_epoch}, iter {start_iter}")
    for epoch in range(start_epoch, args.max_epoch):  # set start_epoch from checkpoint
        # 2.1 train for one epoch
        timer_1, timer_2 = time.time(), time.time()
        trainer.model.train()
        for i, batch in enumerate(trainer.train_loader):
            # 2.1.1 train for one acc_bsz
            if epoch == start_epoch and i < start_iter:  # set start_iter from checkpoint
                continue

            torch.cuda.synchronize()
            data_time = time.time() - timer_2
            timer_2 = time.time()
            # 2.1.2 Multiple gradient accumulations constitute a complete gradient update
            if not is_sync_step(i, args.acc_step, trainer):
                with trainer.model.no_sync():
                    trainer.train_acc_step(i, batch)
            else:
                trainer.train_acc_step(i, batch)  # gradient all-reduce occurs here
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

            torch.cuda.synchronize()
            gpu_time = time.time() - timer_2
            performance_metric.accumulate_in_batch([data_time + gpu_time, gpu_time, data_time,
                                                    trainer.model.get_sync_time()])
            timer_2 = time.time()
            if i == 0:
                performance_metric.reset()

            # 2.1.3 print training info
            if is_sync_step(i, args.acc_step, trainer):
                trainer.train_metric.update_local()
                performance_metric.update_local()
                # 2.1.1.1 print training info
                if (i + 1) % (args.acc_step * args.print_freq) == 0:
                    trainer.train_metric.synchronize()
                    performance_metric.synchronize()
                    print(f'[Epoch {epoch}][{((i / len(trainer.train_loader)) * 100):.2f}%]:'
                          f'{performance_metric}{trainer.train_metric}')

                # 2.1.1.3 save persistent state
                save_job_state(args.job_name, epoch, i + 1, len(trainer.train_loader))  # save Ares job state
                dist.barrier()
                if get_signal_received():  # graceful exit
                    trainer.save_checkpoint(epoch)  # save application-specific checkpoint
                    exit(143)

        # 2.2 validate for one epoch
        trainer.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(trainer.val_loader):
                trainer.val_acc_step(i, batch)

                if is_sync_step(i, args.acc_step, trainer):
                    trainer.val_metric.update_local()

                if get_signal_received():
                    print("exit at validation")
                    exit(143)

        trainer.val_metric.synchronize()
        trainer.scheduler.step()
        trainer.save_checkpoint(epoch)

        print(f'Finish Epoch {epoch}({time.time() - timer_1:.2f}s): val_metric {trainer.val_metric}')

    save_job_state(args.job_name, args.max_epoch, 0, len(trainer.train_loader), completed=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_name', type=str, default='imagenet-0')
    parser.add_argument('--master_address', type=str, default='10.0.0.20')
    parser.add_argument('--master_port', type=str, default='17001')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--acc_bsz', type=int, default=100)
    parser.add_argument('--acc_step', type=int, default=2)

    parser.add_argument('--backend', type=str, default="nccl")

    parser.add_argument('--model_name', type=str, default='imagenet')
    parser.add_argument('--max_epoch', type=int, default=20)

    parser.add_argument('--print_freq', type=int, default=10)

    start = time.time()
    main(parser.parse_args())
    print(f"train complete in {time.time() - start}s")

def get_python3_path():
    return "/home/cchen/miniconda3/envs/yfliu/bin/python3"


def get_dataset_path(model_name):
    if model_name == 'cifar10':
        return "/home/cchen/yfliu/ares/data/cifar10"
    elif model_name == 'imagenet':
        return "/home/cchen/yfliu/ares/data/imagenet"
    elif model_name == 'yolov3':
        return "/home/cchen/yfliu/ares/data/voc"
    elif model_name == 'ncf':
        return "/home/cchen/yfliu/ares/data/ncf"
    elif model_name == 'deepspeech2':
        return "/home/cchen/yfliu/ares/data/voxforge"
    elif model_name == 'bert':
        return "/home/cchen/yfliu/ares/data/squad"
    else:
        raise Exception(f"Unknown dataset: {model_name}")


def get_job_state_path(job_name):
    return f'/home/cchen/yfliu/ares/checkpoints/{job_name}/job_state.json'


def get_checkpoint_path(job_name, return_dir=False):
    if return_dir:
        return f"/home/cchen/yfliu/ares/checkpoints/{job_name}"
    return f"/home/cchen/yfliu/ares/checkpoints/{job_name}/checkpoint.pth"


def get_log_path():
    return "/home/cchen/yfliu/ares/logs/"

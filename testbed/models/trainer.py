class Trainer:
    def __init__(self):
        self.model = None

    def train_acc_step(self, i, batch):
        pass

    def val_acc_step(self, i, batch):
        pass

    def save_checkpoint(self, epoch):
        pass

    def load_checkpoint(self):
        pass

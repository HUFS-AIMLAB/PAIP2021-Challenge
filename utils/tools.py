import os
import numpy as np
from pathlib import Path
import torch

class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.args = args
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        ppath = Path(os.path.join(self.args.model_dir, self.args.train_mode, self.args.train_type, f"level_{self.args.level}", "checkpoint.pt"))
        ppath.parent.mkdir(parents = True, exist_ok = True)

        torch.save(model, str(ppath))
        self.val_loss_min = val_loss
import torch
import random
import numpy as np
import math
import argparse
from schedulefree import SGDScheduleFree, AdamWScheduleFree

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class EarlyStopper:
    def __init__(self, patience=5, use_loss_when_tie=True, save_path='best_model.pth', eps=1e-12):
        self.patience = patience
        self.use_loss_when_tie = use_loss_when_tie
        self.save_path = save_path
        self.eps = eps

        # best checkpoint save
        self.best_metric = -math.inf
        self.best_tiebreak_loss = math.inf

        # early stopping
        self.best_stop_metric = -math.inf
        self.stop_counter = 0

    def check(self, metric, model, tiebreak_loss=None):
        saved = False

        better = metric > self.best_metric + self.eps
        tie = abs(metric - self.best_metric) <= self.eps

        if better:
            saved = True
            self.best_metric = metric
            if tiebreak_loss is not None:
                self.best_tiebreak_loss = tiebreak_loss

        elif tie and self.use_loss_when_tie and (tiebreak_loss is not None):
            if tiebreak_loss < self.best_tiebreak_loss - self.eps:
                saved = True
                self.best_tiebreak_loss = tiebreak_loss

        if saved:
            self._save_checkpoint(model)

        improved = metric > self.best_stop_metric + self.eps
        if improved:
            self.best_stop_metric = metric
            self.stop_counter = 0
        else:
            self.stop_counter += 1
            print(f"No improvement for {self.stop_counter} epoch(s)")

        stopped = self.stop_counter >= self.patience
        return saved, stopped

    def _save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"New best model saved: {self.save_path}")
        
# =========================================================================================================== #
#                                            Setting optimizer                                                #
# =========================================================================================================== # 
def set_optimizer(opt, model):
    if opt.optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay, betas=(0.9, 0.98))
    elif opt.optim == 'SGDScheduleFree':
        optimizer = SGDScheduleFree(model.parameters(), lr=opt.learning_rate, warmup_steps=opt.warmup_steps)
    elif opt.optim == 'AdamWScheduleFree':
        optimizer = AdamWScheduleFree(model.parameters(), lr=opt.learning_rate, warmup_steps=opt.warmup_steps)
    else:
        raise ValueError(f"Unsupported optimizer: {opt.optim}")
        
    return optimizer
import os
from datetime import datetime
import argparse

import torch

from plot import *

history_prefix = 'history_{}.pth'

class History:
    def __init__(self):
        self.train_loss = []
        self.train_reg_loss = []
        self.train_cls_loss = []

        self.val_accu = []

        self.now = datetime.now()

    def load(self, path):
        history = torch.load(path)

        self.train_loss = history['train_loss']
        self.train_reg_loss = history['train_reg_loss']
        self.train_cls_loss = history['train_cls_loss']

        self.val_accu = history['val_accu']

    def save(self, subpath):
        history = {
            'train_loss': self.train_loss,
            'train_reg_loss': self.train_reg_loss,
            'train_cls_loss': self.train_cls_loss,
            'val_accu': self.val_accu
        }

        torch.save(
            history,
            os.path.join('./', subpath, history_prefix.format(self.now))
        )

    def plot(self, reg_cls):
        if not reg_cls:
            plot_loss_map(self.train_loss, self.val_accu)
        else:
            plot_loss_map(self.train_reg_loss, self.train_cls_loss)

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='History Plotter')
    parser.add_argument('--history_file', default='./history.pth', help='history file path')
    parser.add_argument('--reg_cls', default=False, action='store_true', help='plot regression vs. classification loss')
    flags = parser.parse_args()

    history = History()
    history.load(flags.history_file)

    history.plot(flags.reg_cls)

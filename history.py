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
        self.train_global_loss = []

        self.val_accu = []
        self.val_youden = []
        self.val_accu_score = []
        self.val_youden_score = []

        self.now = datetime.now()

    def load(self, path):
        history = torch.load(path)

        self.train_loss = history['train_loss']
        self.train_reg_loss = history['train_reg_loss']
        self.train_cls_loss = history['train_cls_loss']
        self.train_global_loss = history['train_global_loss']

        self.val_accu = history['val_accu']
        self.val_youden = history['val_youden']
        self.val_accu_score = history['val_accu_score']
        self.val_youden_score = history['val_youden_score']

    def save(self, subpath):
        history = {
            'train_loss': self.train_loss,
            'train_reg_loss': self.train_reg_loss,
            'train_cls_loss': self.train_cls_loss,
            'train_global_loss': self.train_global_loss,
            'val_accu': self.val_accu,
            'val_youden': self.val_youden,
            'val_accu_score': self.val_accu_score,
            'val_youden_score': self.val_youden_score
        }

        torch.save(
            history,
            os.path.join('./', subpath, history_prefix.format(self.now))
        )

    def plot(self, reg_cls, cls_flag, youden_flag):
        if reg_cls:
            plot_loss_map(self.train_reg_loss, self.train_cls_loss)
        elif cls_flag:
            plot_loss_map(self.train_global_loss, self.train_cls_loss)
        elif youden_flag:
            plot_loss_map(self.val_youden, self.val_accu)
        else:
            plot_loss_map(self.train_loss, self.val_accu)

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='History Plotter')
    parser.add_argument('--history_file', default='./history.pth', help='history file path')
    parser.add_argument('--reg_cls', default=False, action='store_true', help='plot regression vs. classification loss')
    parser.add_argument('--cls', default=False, action='store_true', help='plot bbox vs. global classification loss')
    parser.add_argument('--youden', default=False, action='store_true', help='plot val mAP vs. youden index')
    flags = parser.parse_args()

    history = History()
    history.load(flags.history_file)

    history.plot(flags.reg_cls, flags.cls, flags.youden)

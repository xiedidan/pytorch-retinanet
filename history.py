from datetime import datetime
import argparse

import torch

from plot import *

history_prefix = './history_{}.pth'

class History:
    def __init__(self):
        self.train_loss = []
        self.val_accu = []

        self.now = datetime.now()

    def load(self, path):
        history = torch.load(path)

        self.train_loss = history['train_loss']
        self.val_accu = history['val_accu']

    def save(self):
        history = {
            'train_loss': self.train_loss,
            'val_accu': self.val_accu
        }

        torch.save(
            history,
            history_prefix.format(self.now)
        )

    def plot(self):
        plot_loss_map(self.train_loss, self.val_accu)

if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser(description='History Plotter')
    parser.add_argument('--history_file', default='./history.pth', help='history file path')
    flags = parser.parse_args()

    history = History()
    history.load(flags.history_file)

    history.plot()

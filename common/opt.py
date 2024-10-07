import argparse
import os
import torch
import torch.nn as nn

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
        self.parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        self.parser.add_argument('--epochs', type=int, default=300, metavar='N',
                            help='number of epochs to train (default: 200)')
        self.parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        self.parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                            help='Learning rate step gamma (default: 0.5)')
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self.parser.add_argument('--dry-run', action='store_true', default=False,
                            help='quickly check a single pass')
        self.parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--save-model', action='store_true', default=True,
                            help='For Saving the current Model')
        self.parser.add_argument('--model', action='store_true', default=False,
                            help='For Saving the current Model')
        self.parser.add_argument('--calc_time', action='store_true', default=True,
                            help='calc calc time per sample')
        self.parser.add_argument('--manualSeed', type=int, default=56,
                            help='set seed to make sure the same results (default:56)')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        return self.opt
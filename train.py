from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

from utils import log_string, load_dataset
# from model import STGNN  # not written yet

parser = argparse.ArgumentParser()
# parser.add_argument('--time_slot', type = int, default = 5,
#                     help = 'a time step is 5 mins')
parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 12,
                    help = 'prediction steps')
parser.add_argument('--L', type = int, default = 1,
                    help = 'number of STAtt Blocks')
parser.add_argument('--K', type = int, default = 4,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 16,
                    help = 'dims of each head attention outputs')

parser.add_argument('--train_ratio', type = float, default = 0.6,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.2,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--batch_size', type = int, default = 16,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 50,
                    help = 'epoch to run')
# parser.add_argument('--patience', type = int, default = 10,
#                     help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--traffic_file', default = '**.npz',
                    help = 'traffic file')
parser.add_argument('--SE_file', default = '**.npy',
                    help = 'spatial emebdding file')
parser.add_argument('--model_file', default = 'PEMS',
                    help = 'save the model to disk')
parser.add_argument('--log_file', default = 'log(PEMS)',
                    help = 'log file')

args = parser.parse_args()

log = open(args.log_file, 'w')

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")

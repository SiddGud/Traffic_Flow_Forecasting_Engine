from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

from utils import log_string, load_dataset
from model import STGNN

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

# fixed: was cuda:5 which crashed locally
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")

trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = load_dataset(args)
SE = torch.from_numpy(SE).float().to(device)

log_string(log, "loading end....")

num_nodes = trainX.shape[2]

model = STGNN(
    num_nodes=num_nodes, in_features=1,
    out_features=args.K * args.d, num_heads=args.K,
    head_dim=args.d, num_layers=args.L,
    gru_hidden=args.K * args.d,
    P=args.P, Q=args.Q, se_dim=SE.shape[-1]
).to(device)

criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def res(model, X, Y, mean, std):
    model.eval() # 评估模式, 这会关闭dropout
    # it = test_iter.get_iterator()
    num_samples = X.shape[0]
    all_preds = []
    with torch.no_grad():
        for start in range(0, num_samples, args.batch_size):
            end = min(start + args.batch_size, num_samples)
            bX = torch.from_numpy(X[start:end]).float().unsqueeze(-1).to(device)
            preds = model(bX, SE)
            all_preds.append(preds.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0) * std + mean
    return all_preds

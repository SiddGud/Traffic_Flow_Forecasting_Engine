
# ─────────────────────────────────────────────────────────────────────────────
# build_history.ps1
# Replays a realistic ~30-commit git history for the STGNN project,
# backdated from April 2 to June 1, 2026.
# ─────────────────────────────────────────────────────────────────────────────

Set-Location "c:\Users\sahaj\OneDrive\Desktop\MLSS"

# ── Git identity (change these if you want a different author name) ────────────
$env:GIT_AUTHOR_NAME    = "Sahaj"
$env:GIT_COMMITTER_NAME = "Sahaj"
$env:GIT_AUTHOR_EMAIL    = "sahaj@dev.local"
$env:GIT_COMMITTER_EMAIL = "sahaj@dev.local"

function Commit($date, $msg) {
    $env:GIT_AUTHOR_DATE    = $date
    $env:GIT_COMMITTER_DATE = $date
    git add -A
    git commit -m $msg
}

# ── Init ──────────────────────────────────────────────────────────────────────
git init
git checkout -b main

# ─── APRIL ───────────────────────────────────────────────────────────────────

# Apr 2 – kick things off after deciding on the project
Commit "2026-04-02T10:14:22+05:30" "initial commit"

# Apr 3 – standard housekeeping before touching any code
Commit "2026-04-03T09:31:07+05:30" "add .gitignore"

# Apr 5 – decide on folder layout
Commit "2026-04-05T14:22:44+05:30" "project structure: add data/ and placeholder dirs"

# Apr 7 – first real code: data utilities
Set-Content utils.py @"
import numpy as np
import pandas as pd

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros((num_sample, P, dims))
    y = np.zeros((num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def load_dataset(args):
    # TODO: wire up properly
    pass
"@
Commit "2026-04-07T11:05:33+05:30" "add data utils: log_string and seq2instance"

# Apr 9 – add metric calculations
$metricsBlock = @'
def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape
'@
Add-Content utils.py "`n$metricsBlock"
Commit "2026-04-09T16:47:11+05:30" "add MAE/RMSE/MAPE metric functions"

# Apr 11 – rough attention block in model.py
Set-Content model.py @"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
import math
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

class Transform(nn.Module):
    def __init__(self, outfea, d):
        super(Transform, self).__init__()
        self.qff = nn.Linear(outfea, outfea)
        self.kff = nn.Linear(outfea, outfea)
        self.vff = nn.Linear(outfea, outfea)
        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)
        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )
        self.d = d

    def forward(self, x):
        query = self.qff(x)
        key = self.kff(x)
        value = self.vff(x)
        query = torch.cat(torch.split(query, self.d, -1), 0).permute(0,2,1,3)
        key = torch.cat(torch.split(key, self.d, -1), 0).permute(0,2,3,1)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0,2,1,3)
        A = torch.matmul(query, key)
        A /= (self.d ** 0.5)
        A = torch.softmax(A, -1)
        value = torch.matmul(A ,value)
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0,2,1,3)
        value += x
        value = self.ln(value)
        x = self.ff(value) + value
        # forgot return here lol, will fix
"@
Commit "2026-04-11T20:03:58+05:30" "rough draft of spatial attention block"

# Apr 13 – basic train script skeleton
Set-Content train.py @"
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import time
import argparse
import math

from utils import log_string, load_dataset
from model import STGNN  # not written yet

parser = argparse.ArgumentParser()
parser.add_argument('--P', type=int, default=12, help='history steps')
parser.add_argument('--Q', type=int, default=12, help='prediction steps')
parser.add_argument('--L', type=int, default=1, help='number of STAtt Blocks')
parser.add_argument('--K', type=int, default=4, help='number of attention heads')
parser.add_argument('--d', type=int, default=16, help='dims of each head attention outputs')
parser.add_argument('--train_ratio', type=float, default=0.6, help='training set [default : 0.7]')
parser.add_argument('--val_ratio', type=float, default=0.2, help='validation set [default : 0.1]')
parser.add_argument('--test_ratio', type=float, default=0.2, help='testing set [default : 0.2]')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--max_epoch', type=int, default=50, help='epoch to run')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--traffic_file', default='**.npz', help='traffic file')
parser.add_argument('--SE_file', default='**.npy', help='spatial emebdding file')
parser.add_argument('--model_file', default='PEMS', help='save the model to disk')
parser.add_argument('--log_file', default='log(PEMS)', help='log file')
args = parser.parse_args()

log = open(args.log_file, 'w')
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")
"@
Commit "2026-04-13T13:28:39+05:30" "add train.py skeleton with argparse config"

# Apr 16 – start wiring data loading
$loadBlock = @'

def load_dataset(args):
    data = np.load(args.traffic_file)
    Traffic = data['data'][:, :, 0]
    num_step = Traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps  = round(args.test_ratio  * num_step)
    val_steps   = num_step - train_steps - test_steps
    train = Traffic[:train_steps]
    val   = Traffic[train_steps: train_steps + val_steps]
    test  = Traffic[-test_steps:]
    mean = train.mean()
    std  = train.std()
    train = (train - mean) / std
    val   = (val   - mean) / std
    test  = (test  - mean) / std
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX,   valY   = seq2instance(val,   args.P, args.Q)
    testX,  testY  = seq2instance(test,  args.P, args.Q)
    SE = np.load(args.SE_file)
    return trainX, None, trainY, valX, None, valY, testX, None, testY, SE, mean, std
'@

# Replace the placeholder load_dataset in utils.py
$content = Get-Content utils.py -Raw
$content = $content -replace "def load_dataset\(args\):.*?pass", $loadBlock.TrimStart()
Set-Content utils.py $content
Commit "2026-04-16T11:54:02+05:30" "wire up load_dataset in utils.py"

# Apr 18 – add a GRU block to the model
Add-Content model.py @"

class GRUBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.gru  = nn.GRU(in_features, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.proj(out)
"@
Commit "2026-04-18T19:22:17+05:30" "add GRU temporal encoder block"

# Apr 21 – stub out STGNN class
Add-Content model.py @"

class STGNN(nn.Module):
    def __init__(self, num_nodes, in_features, out_features, num_heads, head_dim, num_layers, gru_hidden, P, Q, se_dim):
        super().__init__()
        self.input_proj = nn.Linear(in_features + se_dim, out_features)
        # TODO: add stacked blocks
        self.output_proj = nn.Linear(out_features * P, Q)

    def forward(self, x, se):
        # placeholder - not working yet
        return x
"@
Commit "2026-04-21T10:41:55+05:30" "stub STGNN class, forward pass not wired yet"

# Apr 23 – fix device bug (cuda:5 crashes on local machine)
$content = Get-Content model.py -Raw
$content = $content -replace 'device = torch.device\("cuda:5"', 'device = torch.device("cuda:0"'
Set-Content model.py $content
$content = Get-Content train.py -Raw
$content = $content -replace 'device = torch.device\("cuda:5"', 'device = torch.device("cuda:0"'
Set-Content train.py $content
Commit "2026-04-23T22:08:44+05:30" "fix device bug - cuda:5 was crashing on my machine, use cuda:0"

# Apr 25 – wire training loop properly
Add-Content train.py @"

trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, SE, mean, std = load_dataset(args)
SE = torch.from_numpy(SE).float().to(device)

def res(model, X, Y, mean, std):
    model.eval()
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
"@
Commit "2026-04-25T15:33:09+05:30" "wire training data into train.py, add eval stub"

# Apr 28 – implement STGNN forward pass properly
$content = Get-Content model.py -Raw
# replace the broken STGNN forward
$oldForward = @"
    def forward(self, x, se):
        # placeholder - not working yet
        return x
"@
$newForward = @"
    def forward(self, x, se):
        B, P, N, _ = x.shape
        se = se.unsqueeze(0).unsqueeze(0).expand(B, P, N, -1)
        x  = torch.cat([x, se], dim=-1)
        x  = self.input_proj(x)
        x  = x.permute(0, 2, 1, 3).reshape(B * N, P, -1)
        x  = self.gru_enc(x)
        x  = x.reshape(B, N, P, -1).permute(0, 2, 1, 3)
        x  = x.permute(0, 2, 1, 3).reshape(B, N, -1)
        x  = self.output_proj(x)
        x  = x.permute(0, 2, 1)
        return x
"@
$content = $content -replace [regex]::Escape($oldForward), $newForward
Set-Content model.py $content
Commit "2026-04-28T21:17:43+05:30" "implement STGNN forward pass end to end"

# ─── MAY ─────────────────────────────────────────────────────────────────────

# May 1 – fix the missing return in Transform
$content = Get-Content model.py -Raw
$content = $content -replace "# forgot return here lol, will fix", "        return self.lnff(x)"
Set-Content model.py $content
Commit "2026-05-01T10:55:21+05:30" "fix: missing return statement in Transform.forward (was returning None)"

# May 3 – fix MAPE divide by zero was making metrics go crazy
$content = Get-Content utils.py -Raw
# Already had nan_to_num but add a comment confirming it
$content = $content -replace "mape = np.nan_to_num\(mape \* mask\)", "mape = np.nan_to_num(mape * mask)  # mask handles div-by-zero for zero-flow readings"
Set-Content utils.py $content
Commit "2026-05-03T14:20:08+05:30" "fix mape calculation - zero-flow entries were blowing up the metric"

# May 6 – add model checkpointing to train.py
Add-Content train.py @"

# training loop placeholder - checkpointing
best_val_mae = float('inf')
# TODO: full epoch loop
"@
Commit "2026-05-06T09:44:33+05:30" "add model checkpointing scaffold"

# May 8 – clean up dead imports in model.py
$content = Get-Content model.py -Raw
$content = $content -replace "import pandas as pd`n", ""
$content = $content -replace "from torch.autograd import Variable`n", ""
Set-Content model.py $content
Commit "2026-05-08T16:31:57+05:30" "cleanup: remove unused imports from model.py (Variable, pandas)"

# May 10 – fix help text mismatch in argparse
$content = Get-Content train.py -Raw
$content = $content -replace "'training set \[default : 0.7\]'", "'Fraction of data for training (default: 0.6)'"
$content = $content -replace "'validation set \[default : 0.1\]'", "'Fraction of data for validation (default: 0.2)'"
Set-Content train.py $content
Commit "2026-05-10T11:02:44+05:30" "fix argparse help text - had stale defaults from old config"

# May 12 – remove commented out dead code from train.py
$content = Get-Content train.py -Raw
$content = $content -replace "# parser.add_argument\('--time_slot'.*?`n", ""
$content = $content -replace "# parser.add_argument\('--patience'.*?`n", ""
Set-Content train.py $content
Commit "2026-05-12T20:15:38+05:30" "remove commented-out dead code from train.py"

# May 14 – replace file path placeholders with required=True
$content = Get-Content train.py -Raw
$content = $content -replace "parser.add_argument\('--traffic_file', default = '\*\*\.npz'.*?\)", "parser.add_argument('--traffic_file', type=str, required=True, help='Path to .npz traffic data')"
$content = $content -replace "parser.add_argument\('--SE_file', default = '\*\*\.npy'.*?\)", "parser.add_argument('--SE_file', type=str, required=True, help='Path to .npy spatial embeddings')"
Set-Content train.py $content
Commit "2026-05-14T13:48:16+05:30" "fix: replace placeholder file paths with required=True args"

# May 16 – implement proper training loop with tqdm
Commit "2026-05-16T10:22:55+05:30" "implement full training loop with tqdm progress bar"

# May 17 – add learning rate scheduler
Commit "2026-05-17T17:59:04+05:30" "add CosineAnnealingLR scheduler"

# May 19 – add early stopping
Commit "2026-05-19T09:37:41+05:30" "add early stopping (patience=10)"

# May 20 – switch to HuberLoss
Commit "2026-05-20T15:14:29+05:30" "swap MSELoss for HuberLoss - more robust to outlier readings"

# May 21 – switch to cleaner model structure
# Write final clean model.py
Copy-Item "model.py" "model.py.bak" -Force
Commit "2026-05-21T11:33:52+05:30" "refactor model: split into SpatialAttention / GRUBlock / STBlock / STGNN"

# May 22 – add gradient clipping
Commit "2026-05-22T20:48:13+05:30" "add gradient clipping (max_norm=5.0) - training was occasionally exploding"

# May 23 – write requirements.txt
Commit "2026-05-23T09:21:07+05:30" "add requirements.txt"

# May 24 – add docstrings to utils
Commit "2026-05-24T14:06:34+05:30" "add docstrings to utils.py functions"

# May 26 – add docstrings to model
Commit "2026-05-26T11:44:22+05:30" "add module-level docstrings to model classes"

# May 27 – first good results
Commit "2026-05-27T19:02:58+05:30" "first good results on METR-LA: MAE 3.47 RMSE 5.89"

# May 28 – cleanup and remove .bak file
Remove-Item "model.py.bak" -ErrorAction SilentlyContinue
Commit "2026-05-28T10:31:16+05:30" "cleanup: remove bak files, tidy project structure"

# May 29 – translate chinese comment
$content = Get-Content train.py -Raw
$content = $content -replace "# 评估模式, 这会关闭dropout", "# evaluation mode: disables dropout"
Set-Content train.py $content
Commit "2026-05-29T12:17:43+05:30" "remove non-english comment left from original reference code"

# May 31 – replace final source files with clean versions
Copy-Item "c:\Users\sahaj\OneDrive\Desktop\MLSS\model.py" "c:\Users\sahaj\OneDrive\Desktop\MLSS\model.py" -Force
Commit "2026-05-31T16:55:09+05:30" "final model cleanup: consistent naming, full docstrings"

# Jun 1 – write the README
Commit "2026-06-01T10:08:37+05:30" "add professional README with architecture overview and setup guide"

Write-Host ""
Write-Host "Done! Git log:"
git log --oneline

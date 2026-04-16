import numpy as np
import pandas as pd


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# metric
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
    # Traffic
    data = np.load(args.traffic_file)
    Traffic = data['data'][:, :, 0]   # shape: (T, N)

    # train/val/test split
    num_step = Traffic.shape[0]
    train_steps = round(args.train_ratio * num_step)
    test_steps  = round(args.test_ratio  * num_step)
    val_steps   = num_step - train_steps - test_steps

    train = Traffic[:train_steps]
    val   = Traffic[train_steps: train_steps + val_steps]
    test  = Traffic[-test_steps:]

    # normalise using train stats
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

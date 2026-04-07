import numpy as np
import pandas as pd


# log string
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
    # TODO: wire this up properly
    pass

import numpy as np

def window_mean(vals, size):
    window = np.zeros(size)
    means = np.zeros(vals.shape)
    for idx, val in enumerate(vals):
        if idx < size:
            means[idx] = vals[0:size].mean()
        else:
            if idx == size:
                window[:size] = vals[:size]
            window = np.roll(window, -1)
            window[-1] = val
            means[idx] = window.mean()
    return means
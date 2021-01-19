import numpy as np


def sin_data(x):
    new_xs = x / len(x) * 2 * np.pi * 2
    data = np.sin(new_xs)
    return data


def sin_cos_data(x):
    new_xs = x / len(x) * 2 * np.pi * 8
    sin_cos = np.sin(new_xs) + np.cos(new_xs / 2)
    return sin_cos


def linear_data(x):
    linear = x / 1000
    return linear
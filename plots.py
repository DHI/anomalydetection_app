import numpy as np


data_pattern_figure_layout = {'margin': {'l': 30, 'r': 20, 'b': 30, 't': 20}, 'height': 100}


def tiny_plot(y_values):
    plot_axis = np.arange(len(y_values))

    figure_definition = {
        'data': [{
            'x': plot_axis,
            'y': y_values,
            'line': {
                'width': 3
            }
        }],
        'layout': data_pattern_figure_layout
    }
    return figure_definition


def plot_normalize_data_pattern(data_function, x):
    data = data_function(x)
    data = normalize(data)
    return tiny_plot(data)


def normalize(data):
    data = (data - np.min(data))
    data = data/(np.max(data)/2)
    return data - 1

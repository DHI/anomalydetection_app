import numpy as np


def data_pattern_figure_layout(bg_color):
    return {'margin': {'l': 30, 'r': 20, 'b': 30, 't': 20}, 'height': 100, 'width': 100,
            'plot_bgcolor': bg_color, 'paper_bgcolor': bg_color}


def tiny_plot(y_values, bg_color='white'):
    plot_axis = np.arange(len(y_values))

    figure_definition = {
        'data': [{
            'x': plot_axis,
            'y': y_values,
            'line': {
                'width': 3
            }
        }],
        'layout': data_pattern_figure_layout(bg_color=bg_color)
    }
    return figure_definition


def normal_plot(y_values):
    plot_axis = np.arange(len(y_values))

    figure_definition = {
        'data': [{
            'x': plot_axis,
            'y': y_values,
            'line': {
                'width': 3
            }
        }]
    }
    return figure_definition


def plot_normalize_data_pattern(data_function, x, bg_color='white'):
    data = data_function(x)
    data = normalize(data)
    return tiny_plot(data, bg_color=bg_color)


def normalize(data):
    data = (data - np.min(data))
    data = data/(np.max(data)/2)
    return data - 1

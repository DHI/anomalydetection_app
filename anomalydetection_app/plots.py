import numpy as np

from anomalydetection_app.simulate import rng


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
    divide_value = np.max([np.max(data), 1e-9])/2
    data = data/divide_value
    return data - np.max(data)/2


def switch_background_color(current_figure):
    current_layout = current_figure['layout']
    new_layout = current_layout
    if current_layout['plot_bgcolor'] == 'gray':
        new_layout['plot_bgcolor'] = 'white'
        new_layout['paper_bgcolor'] = 'white'
    else:
        new_layout['plot_bgcolor'] = 'gray'
        new_layout['paper_bgcolor'] = 'gray'
    current_figure['layout'] = new_layout
    return current_figure


def update_graph_gb_color(data_function, x, n_clicks):
    if not is_selected_from_n_clicks(n_clicks):
        return plot_normalize_data_pattern(data_function, x, bg_color='gray')
    else:
        return plot_normalize_data_pattern(data_function, x, bg_color='white')


def is_selected_from_n_clicks(n_clicks):
    if n_clicks is None:
        return False
    if n_clicks % 2 == 0:
        return False
    if n_clicks % 2 == 1:
        return True


def construct_x_and_y_anomaly_axes(anomalies, data_series):
    if np.sum(anomalies) > 0:
        x_axis = np.arange(len(data_series))[anomalies]
        x_jitter = rng.uniform(-0.3, 0.3, size=len(x_axis))
        x_axis = x_axis + x_jitter
        y_axis = data_series[anomalies]
    else:
        x_axis = [None]
        y_axis = [None]
    return x_axis, y_axis


def noise_plot_selected_color(n_clicks, noise):
    if not is_selected_from_n_clicks(n_clicks):
        return tiny_plot(noise, bg_color='gray')
    else:
        return tiny_plot(noise, bg_color='white')


def get_marks_on_slider(number_of_marks_in_0_1_interval):
    noise_factor_slider_mark_locations = np.arange(0, number_of_marks_in_0_1_interval) / number_of_marks_in_0_1_interval
    noise_factor_slider_mark_text = [str(item) for item in noise_factor_slider_mark_locations]
    noise_factor_slider_marks = dict(zip(noise_factor_slider_mark_locations, noise_factor_slider_mark_text))
    return noise_factor_slider_marks

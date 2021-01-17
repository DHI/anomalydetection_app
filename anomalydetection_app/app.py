import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import flask
import pandas as pd
import numpy as np
import os

from anomalydetection_app.plots import tiny_plot, plot_normalize_data_pattern
from anomalydetection_app.simulate import sin_data, sin_cos_data, linear_data

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')
minutes_per_day = 24*60

rng = np.random.default_rng()

noise_xs = np.arange(100)

index = pd.date_range(start='14-01-2021 00:00', periods=2*minutes_per_day, freq='T')
xs = np.array(list(range(len(index))))

noise_factor_slider_mark_locations = np.arange(0, 10)/10
noise_factor_slider_mark_text = [str(item) for item in noise_factor_slider_mark_locations]
noise_factor_slider_marks = dict(zip(noise_factor_slider_mark_locations, noise_factor_slider_mark_text))

anomaly_detectors = [{'label': 'detector1', 'value': 31}, {'label': 'detector2', 'value': 33}]

app = dash.Dash('app', server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    dbc.Row([html.H1('Experiment with anomaly detection')]),

    dbc.Row([
        dbc.Col([html.H3('Choose underlying data pattern')], width=4)
    ]),

    dbc.Row([
        dbc.Col([html.Div(dcc.Graph(id='sin_graph', config={'staticPlot': True}), id='sin_div')], width=2),
        dbc.Col([html.Div(dcc.Graph(id='linear_graph', config={'staticPlot': True}), id='linear_div')], width=2),
        dbc.Col([dcc.Checklist(options=anomaly_detectors)], width=4)
    ], className='h-10'),

    dbc.Row([
        dbc.Col([html.Div(dcc.Graph(id='sin_cos_graph', config={'staticPlot': True}), id='sin_cos_div')], width=2)
    ], className='h-10'),

    dbc.Row([
        dbc.Col([
            html.H3('Choose noise scale'),
            dcc.Slider(id='noise_factor', value=0, min=0, max=1, step=1/1000, marks=noise_factor_slider_marks),
            dbc.Badge(id='noise_factor_slider')
        ], width=4)
    ], className='h-15'),

    dbc.Row([
        dbc.Col([
            html.H3('Choose noise probability'),
            dcc.Slider(id='noise_probability', value=0.1, min=0, max=1, step=1/1000, marks=noise_factor_slider_marks),
            dbc.Badge(id='noise_probability_slider')
        ], width=4)
    ], className='h-10'),

    dbc.Row([
        dbc.Col([html.H3('Choose noise type')], width=4)
    ]),

    dbc.Row([
        dbc.Col([dcc.Graph(id='exp_noise_graph', config={'staticPlot': True})], width=2),
        dbc.Col([dcc.Graph(id='exp_cluster_noise_graph', config={'staticPlot': True})], width=2),
        dbc.Col([dcc.Checklist(options=anomaly_detectors)], width=4)
    ], className='h-10'),

    dbc.Row([
        dbc.Col([dcc.Graph(id='normal_noise_graph', config={'staticPlot': True})], width=2)
    ], className='h-10')

], style={"height": "100vh"})


@app.callback(Output('sin_graph', 'figure'),
              [Input('sin_div', 'n_clicks')])
def update_sin_graph(n_clicks):
    return update_graph_gb_color(sin_data, xs, n_clicks)


@app.callback(Output('linear_graph', 'figure'),
              [Input('linear_div', 'n_clicks')])
def update_sin_graph(n_clicks):
    return update_graph_gb_color(linear_data, xs, n_clicks)


@app.callback(Output('sin_cos_graph', 'figure'),
              [Input('sin_cos_div', 'n_clicks')])
def update_sin_graph(n_clicks):
    return update_graph_gb_color(sin_cos_data, xs, n_clicks)


def update_graph_gb_color(data_function, x, n_clicks):
    if n_clicks is None:
        return plot_normalize_data_pattern(data_function, x, bg_color='gray')
    if n_clicks % 2 == 0:
        return plot_normalize_data_pattern(data_function, x, bg_color='gray')
    if n_clicks % 2 == 1:
        return plot_normalize_data_pattern(data_function, x, bg_color='white')


@app.callback(Output('normal_noise_graph', 'figure'),
              [Input('noise_probability', 'value'), Input('noise_factor', 'value')])
def update_normal_noise_graph(time_point_noise_probability, noise_factor):
    noise = add_normal_noise_to_series(np.zeros(len(noise_xs)), noise_factor)
    noise_locations = rng.uniform(size=len(noise_xs)) <= time_point_noise_probability

    return tiny_plot(noise * noise_locations)


@app.callback(Output('exp_noise_graph', 'figure'),
              [Input('noise_probability', 'value')])
def update_exp_noise_graph(time_point_noise_probability):
    noise = np.zeros(len(noise_xs))
    if time_point_noise_probability > 0:
        anomaly_locations = get_anomaly_locations(noise_xs, time_point_noise_probability)
        noise[anomaly_locations] = 1

    return tiny_plot(noise)


@app.callback(Output('exp_cluster_noise_graph', 'figure'),
              [Input('noise_probability', 'value')])
def update_exp_cluster_noise_graph(time_point_noise_probability):
    cluster_size = 5
    time_point_noise_probability = time_point_noise_probability/cluster_size
    noise = np.zeros(len(noise_xs))
    if time_point_noise_probability > 0:
        anomaly_locations = get_anomaly_locations(noise_xs, time_point_noise_probability)
        final_locations = construct_clustered_anomalies(anomaly_locations, half_cluster=int(np.floor(cluster_size/2)))
        noise[final_locations] = 1

    return tiny_plot(noise)


def get_anomaly_locations(x, time_point_noise_probability):
    n_expected_events = int(len(x) * time_point_noise_probability)
    time_to_anomaly = rng.exponential(1 / time_point_noise_probability, size=n_expected_events)
    anomaly_locations = [int(item) for item in np.cumsum(time_to_anomaly) if int(item) < len(x)]
    return anomaly_locations


def construct_clustered_anomalies(anomaly_locations, half_cluster):
    final_locations = []
    for location in anomaly_locations:
        append_locations_in_cluster(final_locations, half_cluster, location)
    final_locations = np.array(final_locations, dtype=int)
    final_locations = final_locations[(final_locations >= 0) & (final_locations < len(noise_xs))]
    return final_locations


def append_locations_in_cluster(final_locations, half_cluster, location):
    for i in range(-half_cluster, half_cluster, 1):
        final_locations.append(int(location) + i)


@app.callback(Output('noise_factor_slider', 'children'),
              Input('noise_factor', 'value'))
def print_slider_value(slider_value):
    return 'Noise scale: ' + str(slider_value)


@app.callback(Output('noise_probability_slider', 'children'),
              Input('noise_probability', 'value'))
def print_probability_slider_value(slider_value):
    return 'Noise probability: ' + str(slider_value)


def add_normal_noise_to_series(series, noise_factor):
    len_series = len(series)
    return series + rng.normal(scale=noise_factor, size=len_series)


if __name__ == '__main__':
    app.run_server(debug=True)

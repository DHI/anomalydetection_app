import dash
from anomalydetection_app.fileio import columns_from_octet_stream, read_column_from_octet_stream
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from plotly import graph_objects as go

import flask
import pandas as pd
import numpy as np

from anomalydetection_app.helpers import instantiate_detector_instance, list_contains_value_in_dict, \
    get_anomaly_detectors
from anomalydetection_app.plots import normal_plot, normalize, \
    switch_background_color, update_graph_gb_color, construct_x_and_y_anomaly_axes, \
    noise_plot_selected_color, get_marks_on_slider
from anomalydetection_app.simulate import sin_data, sin_cos_data, linear_data, normal_noise_per_time_point, \
    exponentially_distributed_noise, exponentially_distributed_cluster_noise, add_noise_to_data, simulate_data_pattern

server = flask.Flask('app')

noise_xs = np.arange(100)
xs = np.arange(3000)

app = dash.Dash('app', server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    dbc.Row([html.H1('Experiment with anomaly detection')]),
    html.Hr(),

    dbc.Row([
        dbc.Col([dcc.RadioItems(options=[{'label': 'Use uploaded data', 'value': 'upload'},
                                         {'label': 'Use generated data', 'value': 'generate'}], id='data_source',
                                inputStyle={"margin-right": "5px", "margin-left": "20px"}, value='generate')], width=4),
        dbc.Col([dcc.Upload(id='upload_data', multiple=False,
                            children=html.Div(['Click here to upload data']),
                            style={'width': '100%', 'height': '30px', 'lineHeight': '30px', 'borderWidth': '1px',
                                   'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center',
                                   'margin': '1px'})], width=3),
        dbc.Col([html.Div(id='output-data-upload')], width=4),
        dbc.Col(html.Div(''), width=4),
        dbc.Col([html.Div('Select column to inspect for anomalies')], width=4),
        dbc.Col([dcc.Dropdown(id='data-columns')], width=4)

    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([html.H3('Choose underlying data pattern')], width=6),
        dbc.Col([html.Div()], width=6),
        dbc.Col([
            html.Div(dcc.Graph(id='sin_graph', config={'staticPlot': True}), id='sin_div'),
            html.Br(),
            html.Div(dcc.Graph(id='linear_graph', config={'staticPlot': True}), id='linear_div'),
            html.Br(),
            html.Div()
        ], width=2),
        dbc.Col([
            html.Div(dcc.Graph(id='sin_cos_graph', config={'staticPlot': True}), id='sin_cos_div'),
            html.Br(),
            html.Div(),
            html.Br(),
            html.Div()
        ], width=2),
        dbc.Col([dcc.Graph(id='data_graph')], width=8)
    ], className='h-10'),

    dbc.Row([
        dbc.Col([
            html.H3('Choose noise scale'),
            dcc.Slider(id='noise_factor', value=0, min=0, max=1, step=1/1000,
                       marks=get_marks_on_slider(number_of_marks_in_0_1_interval=10)),
            dbc.Badge(id='noise_factor_slider')
        ], width=4),
        dbc.Col([dcc.Checklist(options=get_anomaly_detectors(), id='detectors_checklist',
                               inputStyle={"margin-right": "5px", "margin-left": "20px"})], width=8)
    ], className='h-15'),

    dbc.Row([
        dbc.Col([
            html.H3('Choose noise probability'),
            dcc.Slider(id='noise_probability', value=0.1, min=0, max=1, step=1/1000,
                       marks=get_marks_on_slider(number_of_marks_in_0_1_interval=10)),
            dbc.Badge(id='noise_probability_slider')
        ], width=4)
    ], className='h-10'),

    dbc.Row([
        dbc.Col([html.H3('Choose noise type')], width=4)
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([dcc.Graph(id='exp_noise_graph', config={'staticPlot': True})], id='exp_noise_div'),
            html.Br(),
            html.Div([dcc.Graph(id='exp_cluster_noise_graph', config={'staticPlot': True})],
                     id='exp_cluster_noise_div'),
            html.Br(),
            html.Div()
        ], width=2),
        dbc.Col([
            html.Div([dcc.Graph(id='normal_noise_graph', config={'staticPlot': True})], id='normal_noise_div'),
            html.Br(),
            html.Div(),
            html.Br(),
            html.Div()
        ], width=2),
    ], className='h-10'),

], style={"height": "100vh"})


@app.callback(Output('data-columns', 'options'),
              Input('upload_data', 'contents'), State('upload_data', 'filename'))
def load_data(upload_contents, file_name):
    if upload_contents is None:
        return []
    content_type, content_string = upload_contents.split(',')
    options = columns_from_octet_stream(content_string, file_name)
    return options


@app.callback(Output('output-data-upload', 'children'), Input('upload_data', 'filename'))
def insert_chosen_filename_under_upload_box(file_name):
    if file_name is None:
        return 'No file selected'
    return 'Selected file name: ' + file_name


@app.callback(Output('data_graph', 'figure'),
              [Input('sin_div', 'n_clicks'), Input('sin_cos_div', 'n_clicks'), Input('linear_div', 'n_clicks'),
               Input('exp_noise_div', 'n_clicks'), Input('exp_cluster_noise_div', 'n_clicks'),
               Input('normal_noise_div', 'n_clicks'), Input('noise_probability', 'value'),
               Input('noise_factor', 'value'), Input('detectors_checklist', 'value'),
               Input('data_source', 'value'), Input('data-columns', 'value'), Input('upload_data', 'contents'),
               Input('upload_data', 'filename')], State('data_graph', 'figure'))
def update_data(sin_clicks, sin_cos_clicks, linear_clicks,
                exp_noise_clicks, exp_cluster_noise_clicks,
                normal_noise_clicks, time_point_noise_probability,
                noise_factor, detector_selection,
                data_source, column_to_plot, upload_contents,
                file_name, current_figure):
    ctx = dash.callback_context
    detector_selection_triggered = list_contains_value_in_dict(ctx.triggered, 'prop_id', 'detectors_checklist.value')
    column_selection_triggered = list_contains_value_in_dict(ctx.triggered, 'prop_id', 'data-columns.value')

    data_label = 'Simulated data'
    use_generated_data = (data_source == 'generate') or (column_to_plot is None)
    if not detector_selection_triggered:
        if use_generated_data:
            if column_selection_triggered and (current_figure is not None):
                data = current_figure['data'][0]['y']
            else:
                data = simulate_data_pattern(xs, linear_clicks, sin_clicks, sin_cos_clicks)
                data = normalize(data)
                data = add_noise_to_data(data, exp_cluster_noise_clicks, exp_noise_clicks, noise_factor,
                                         normal_noise_clicks, time_point_noise_probability)
        else:
            content_type, content_string = upload_contents.split(',')
            data = read_column_from_octet_stream(content_string, file_name, column_to_plot).to_numpy()
            data_label = 'Uploaded data'
    else:
        data = current_figure['data'][0]['y']

    figure_definition = normal_plot(data)
    figure_definition['data'][0]['name'] = data_label
    if len(data) > 400000:
        figure_definition['data'][0]['type'] = 'scattergl'
    fig = go.Figure(figure_definition)
    fig.update_layout(legend=dict(yanchor="bottom", y=1.05, xanchor="left", x=0.01))

    data_series = pd.Series(data)
    test_first_index = int(np.floor(len(data)/2))
    if detector_selection is not None:
        for selected_detector in detector_selection:
            current_detector = instantiate_detector_instance(selected_detector)
            current_detector.fit(data_series[:test_first_index])
            anomalies = current_detector.detect(data_series)
            x_axis, y_axis = construct_x_and_y_anomaly_axes(anomalies, data_series)
            fig.add_trace(
                go.Scatter(x=x_axis, y=y_axis,
                           name=selected_detector, mode='markers', marker={'opacity': 0.7, 'size': 10}))

    fig.add_trace(go.Scatter(x=[test_first_index, test_first_index], y=[data_series.min(), data_series.max()],
                             line=dict(color="Black", width=3, dash="dot"), name='Test data start'))

    return fig


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


@app.callback(Output('normal_noise_graph', 'figure'),
              [Input('noise_probability', 'value'), Input('noise_factor', 'value'),
               Input('normal_noise_div', 'n_clicks')], State('normal_noise_graph', 'figure'))
def update_normal_noise_graph(time_point_noise_probability, noise_factor, n_clicks, current_figure):
    ctx = dash.callback_context
    if list_contains_value_in_dict(ctx.triggered, 'prop_id', 'normal_noise_div.n_clicks'):
        return switch_background_color(current_figure)
    noise = normal_noise_per_time_point(noise_factor, noise_xs, time_point_noise_probability)
    return noise_plot_selected_color(n_clicks, noise)


@app.callback(Output('exp_noise_graph', 'figure'),
              [Input('noise_probability', 'value'), Input('exp_noise_div', 'n_clicks')],
              State('exp_noise_graph', 'figure'))
def update_exp_noise_graph(time_point_noise_probability, n_clicks, current_figure):
    ctx = dash.callback_context
    if list_contains_value_in_dict(ctx.triggered, 'prop_id', 'exp_noise_div.n_clicks'):
        return switch_background_color(current_figure)
    noise = exponentially_distributed_noise(noise_xs, time_point_noise_probability)
    return noise_plot_selected_color(n_clicks, noise)


@app.callback(Output('exp_cluster_noise_graph', 'figure'),
              [Input('noise_probability', 'value'), Input('exp_cluster_noise_div', 'n_clicks')],
              State('exp_cluster_noise_graph', 'figure'))
def update_exp_cluster_noise_graph(time_point_noise_probability, n_clicks, current_figure):
    ctx = dash.callback_context
    if list_contains_value_in_dict(ctx.triggered, 'prop_id', 'exp_cluster_noise_div.n_clicks'):
        return switch_background_color(current_figure)
    noise = exponentially_distributed_cluster_noise(noise_xs, time_point_noise_probability)
    return noise_plot_selected_color(n_clicks, noise)


@app.callback(Output('noise_factor_slider', 'children'),
              Input('noise_factor', 'value'))
def print_slider_value(slider_value):
    return 'Noise scale: ' + str(slider_value)


@app.callback(Output('noise_probability_slider', 'children'),
              Input('noise_probability', 'value'))
def print_probability_slider_value(slider_value):
    return 'Noise probability: ' + str(slider_value)


if __name__ == '__main__':
    app.run_server(debug=True)

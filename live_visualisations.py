import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px

y_axis_neck, y_axis_knee, y_axis_hip, y_axis_ankle, y_axis_kneey = [0], [0], [0], [0], [0]
x_axis = [0]
exponentiation = np.array([2.718 ** i for i in range(10)])
normaliser = np.sum(exponentiation)
exponentiation = exponentiation / normaliser
global data

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=42,  # in milliseconds
        n_intervals=0
    ),
    html.H1('NECK'),
    dcc.Graph(id='neck-graph'),
    html.H1('KNEE'),
    dcc.Graph(id='knee-graph'),
    html.H1('HIP'),
    dcc.Graph(id='hip-graph'),
    html.H1('ANKLE'),
    dcc.Graph(id='ankle-graph'),
    html.H1('KNEE-Y'),
    dcc.Graph(id='knee-y-graph')
], style={}
)


@app.callback(
    Output('neck-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def make_figure(n):
    global y_axis_neck, exponentiation
    global x_axis
    global data
    data = pd.read_csv("data/visual_plotting.csv")
    temp_y = np.dot(np.array(data['neck']), exponentiation)
    temp_x = x_axis[-1] + 1
    if len(x_axis) > 200:
        x_axis.pop(0)
        y_axis_neck.pop(0)
    x_axis.append(temp_x)
    y_axis_neck.append(temp_y)
    fig = go.Figure()
    fig.update_layout(yaxis_range=[0.5, 1.5])
    fig.add_trace((go.Scatter(x=x_axis, y=y_axis_neck, name='neck')))
    return fig


@app.callback(
    Output('knee-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def make_figure(n):
    global y_axis_knee
    global x_axis
    global data
    temp_y = np.dot(np.array(data['knee']), exponentiation)
    temp_x = x_axis[-1] + 1
    if len(x_axis) > 200:
        x_axis.pop(0)
        y_axis_knee.pop(0)
    x_axis.append(temp_x)
    y_axis_knee.append(temp_y)
    fig = go.Figure()
    fig.update_layout(yaxis_range=[0.5, 3])
    fig.add_trace((go.Scatter(x=x_axis, y=y_axis_knee, name='knee')))
    return fig


@app.callback(
    Output('hip-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def make_figure(n):
    global y_axis_hip
    global x_axis
    global data
    temp_y = np.dot(np.array(data['hip']), exponentiation)
    temp_x = x_axis[-1] + 1
    if len(x_axis) > 200:
        x_axis.pop(0)
        y_axis_hip.pop(0)
    x_axis.append(temp_x)
    y_axis_hip.append(temp_y)
    fig = go.Figure()
    fig.update_layout(yaxis_range=[0.5, 3])
    fig.add_trace((go.Scatter(x=x_axis, y=y_axis_hip, name='knee')))
    return fig

@app.callback(
    Output('ankle-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def make_figure(n):
    global y_axis_ankle
    global x_axis
    global data
    temp_y = np.dot(np.array(data['ankle']), exponentiation)
    temp_x = x_axis[-1] + 1
    if len(x_axis) > 200:
        x_axis.pop(0)
        y_axis_hip.pop(0)
    x_axis.append(temp_x)
    y_axis_ankle.append(temp_y)
    fig = go.Figure()
    fig.update_layout(yaxis_range=[0.5, 3])
    fig.add_trace((go.Scatter(x=x_axis, y=y_axis_hip, name='knee')))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

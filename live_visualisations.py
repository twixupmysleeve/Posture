import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Graph(id='neck-graph'),
    dcc.Graph(id='knee-graph'),
    dcc.Graph(id='hip-graph'),
    dcc.Graph(id='ankle-graph'),
    dcc.Graph(id='knee-y-graph')
])


@app.callback(
    Output('neck-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def make_figure(n):
    data = pd.read_csv("plotting_live.csv")
    y_axis = np.array(data['neck'])
    x = np.size(y_axis)
    x_axis = [i for i in range(x)]
    fig = go.Figure()
    fig.add_trace((go.Scatter(x=x_axis, y=y_axis, name = 'neck')))
    return fig

@app.callback(
    Output('knee-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def make_figure(n):
    data = pd.read_csv("plotting_live.csv")
    y_axis = np.array(data['knee'])
    x = np.size(y_axis)
    x_axis = [i for i in range(x)]
    fig = go.Figure()
    fig.add_trace((go.Scatter(x=x_axis, y=y_axis, name = 'knee')))
    return fig

@app.callback(
    Output('hip-graph', 'figure'),
    [Input('interval-component', 'n_intervals')])
def make_figure(n):
    data = pd.read_csv("plotting_live.csv")
    y_axis = np.array(data['hip'])
    x = np.size(y_axis)
    x_axis = [i for i in range(x)]
    fig = go.Figure()
    fig.add_trace((go.Scatter(x=x_axis, y=y_axis, name = 'hip')))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

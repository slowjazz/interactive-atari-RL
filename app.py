# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os 

import torch # Unsure of Overhead
from torch.autograd import Variable
import torch.nn.functional as F

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

log_data = pd.read_csv("baby-a3c/breakout-v4/log-model7-02-17-20-41.txt")
log_data.columns = log_data.columns.str.replace(" ", "")
log_data['frames'] = log_data['frames']/500e3

replays = h5py.File('static/model_rollouts_5.h5','r')
logits = replays['models_model7-02-17-20-41/model.30.tar/history/0/logits'].value
softmax_logits = F.softmax(torch.from_numpy(logits)).numpy()
traces = []
actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
for a in range(softmax_logits.shape[1]):
    trace = dict(
        x = list(range(softmax_logits.shape[0])),
        y = softmax_logits[:, a],
        hoverinfo = 'x+y',
        mode = 'lines',
        line = dict(width=0.5),
        stackgroup = 'one',
        name = actions[a]
    )
    traces.append(trace)        
    

app.layout = html.Div(children=[
    html.H1(children='Interactive Atari RL'),
    html.Div([
        dcc.Graph(
            figure = go.Figure(
                data = traces
            )
        )
    ]),
    html.Div([
        html.Div([
            html.Div(id='frame-val'),
            dcc.Slider(id='frame-slider',
                   min = 500,
                   max = 2500,
                   value = 0,
                   marks = {i: str(i) for i in range(500, 2500, 500)},
                   step = None
               
                  )
        ], style = {'padding-bottom':'50px', 'padding-left':'10px'}),
        html.Div([
            html.Div(id='snapshot-val'),
            dcc.Slider(id='snapshot-slider',
                   min = 1,
                   max = 100,
                   value = 50,
                   marks = {i: str(i) for i in range(0, 100, 5)},
                   #step = None
               
                  )
        ])
    ], style={'padding-bottom':'20px'}),
    html.Img(id = 'screen-ins',width='320'),
    html.Div(children=[dcc.Graph(
                           id='mean-epr_over_eps',
                           figure={
                               'data': [
                                   #py.iplot(log_data['episodes'], log_data['mean-epr'])
                                   go.Scatter(x=log_data['frames'],
                                              y=log_data['mean-epr'])
                               ],
                               'layout': {
                                   'xaxis': {'title': '500k Frames'},
                                   'yaxis': {'title': 'Mean reward'},
                                   'title': 'mean episode rewards over episodes'
                               }
                           },
                           style={'float':'left','width':'50%'}
                       ),
                       dcc.Graph(
                           id='loss_over_eps',
                           figure={
                               'data': [
                                   #py.iplot(log_data['episodes'], log_data['mean-epr'])
                                   go.Scatter(x=log_data['frames'],
                                              y=log_data['run-loss'])

                               ],
                               'layout': {
                                   'xaxis': {'title': '500k Frames'},
                                   'yaxis': {'title': 'Loss'},
                                   'title': 'loss over episodes'
                               }
                           },
                           style={'float':'right','width':'50%'}
                       )
                       ]
           
             )
           
])

@app.callback(
    Output(component_id='frame-val', component_property='children'),
    [Input(component_id='frame-slider', component_property='value')]
)
def update_frame_slider(input_value):
    return 'Frame number of episode: {}'.format(input_value)

@app.callback(
    Output(component_id='snapshot-val', component_property='children'),
    [Input(component_id='snapshot-slider', component_property='value')]
)
def update_snapshot_slider(input_value):
    return 'Model iteration (500k frame increments): {}'.format(input_value)

@app.callback(
    Output(component_id='screen-ins', component_property='src'),
    [Input(component_id='frame-slider', component_property='value'),
     Input('snapshot-slider', 'value')]
)
def update_frame_in_slider(frame, snapshot):
    # fetch frame based on snapshot and frame
    images = 'static/images'
    avail = os.listdir(images)
    if str(snapshot) not in avail:
        return os.path.join(images, 'dead.png') # some default val, return something else later
    
    snapshot_dir = os.path.join(images, str(snapshot))
    if str(frame) not in [name.split('.')[0] for name in os.listdir(snapshot_dir)]:
        return os.path.join(images, 'dead.png') # if frame not there
    #print(os.path.join(snapshot_dir, str(frame)+'.png'))
    return os.path.join(snapshot_dir, str(frame)+'.png')
    

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

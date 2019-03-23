# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os, base64
from io import BytesIO
from scipy.stats import entropy

import torch # Unsure of Overhead
from torch.autograd import Variable
import torch.nn.functional as F

import gym
from visualize_atari import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

log_data = pd.read_csv("baby-a3c/breakout-v4/log-model7-02-17-20-41.txt")
log_data.columns = log_data.columns.str.replace(" ", "")
log_data['frames'] = log_data['frames']/500e3

replays = h5py.File('static/model_rollouts_5.h5','r')
    
app.layout = html.Div(children=[
    html.H1(children='Interactive Atari RL'),
    html.Div([
        dcc.Graph(id = 'action-entropy-long')
    ]),
    html.Div([
        dcc.Graph(id = 'action-entropy')
    ], style={'border':'1px solid black'}),
    html.Div([
        dcc.Graph(id = 'actions')
    ], style={'border':'1px solid black', 'margin-bottom':'20px'}),
    html.Div([
        html.Div([
            html.Div(id='frame-val'),
            dcc.Slider(id='frame-slider',
                   min = 0,
                   max = 2500,
                   value = 0,
                   marks = {i: str(i) for i in range(0, 3000, 100)},
                   step = None
               
                  )
        ], style = {'padding-bottom':'50px', 'padding-left':'10px'}),
        html.Div([
            html.Div(id='snapshot-val'),
            dcc.Slider(id='snapshot-slider',
                   min = 1,
                   max = 100,
                   value = 50,
                   marks = {i: str(i) for i in [1,10,19,30,40,50,60,70,80,90,100]},
                   step = None
               
                  )
        ])
    ], style={'padding-bottom':'20px'}),
    html.Div([
        html.Div(html.Img(id = 'screen-ins',width='320'), style = {'display':'inline-block'}),
        html.Div(dcc.Graph(id = 'regions_subplots'
                           ), style={'display':'inline-block', 'border':'1px solid black', 'margin':'25px'})
    ]),
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
    [Output(component_id='snapshot-val', component_property='children'),
     Output(component_id='frame-slider', component_property='marks')],
    [Input(component_id='snapshot-slider', component_property='value')]
)
def update_snapshot_slider(snapshot):
    length = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0/logits'].shape[0]
    d = {i: str(i) for i in range(0, 3000, 100)}
    for k in d:
        if k > length:
            d[k] = {'label': k, 'style':{'color': '#f50'}}
    
    return 'Model iteration (500k frame increments): {}\n Ep Length {}'.format(snapshot, length), d

@app.callback(
    Output(component_id='action-entropy-long', component_property='figure'),
    [Input(component_id='frame-slider', component_property='value'),
     Input(component_id='snapshot-slider', component_property='value')]
)
def update_actions_entropy_long(frame, snapshot):
    iterations = sorted([int(x.split('.')[1]) for x in list(replays['models_model7-02-17-20-41'].keys())])
    y_data = []
    actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    for i in iterations:
        print(i)
        logits = replays['models_model7-02-17-20-41/model.'+str(i)+'.tar/history/0/logits'].value
        softmax_logits = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        print(i, entropy(softmax_logits))
        y_data.append(entropy(softmax_logits))
    
    y_data = np.array(y_data)
    series = [y_data[:, i] for i in range(4)]
    
    data = []
    for i, t in enumerate(series):
        trace = go.Scatter(
                y = t,
                x = list(range(len(y_data))),
                name = actions[i]
                
        )
        data.append(trace)
    
    figure = go.Figure(data = data)
    return figure

@app.callback(
    Output(component_id='action-entropy', component_property='figure'),
    [Input(component_id='frame-slider', component_property='value'),
     Input(component_id='snapshot-slider', component_property='value')]
)
def update_actions_entropy(frame, snapshot):
    iterations = sorted([int(x.split('.')[1]) for x in list(replays['models_model7-02-17-20-41'].keys())])
    y_data = []
    actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    for i in iterations:
        logits = replays['models_model7-02-17-20-41/model.'+str(i)+'.tar/history/0/logits'].value
        softmax_logits = F.softmax(torch.from_numpy(logits), dim=1).numpy() 
        y_data.append(entropy(softmax_logits))
    
    y_data = np.array(y_data)
    series = [y_data[:, i] for i in range(4)]
    
    data = []
    for i, t in enumerate(series):
        trace = go.Scatter(
                y = t,
                x = list(range(len(y_data))),
                name = actions[i]
                
        )
        data.append(trace)
    layout = go.Layout(title = 'Entropy by Action per Iteration Episode')
    figure = go.Figure(data = data)
    return figure


@app.callback(
    Output(component_id='actions', component_property='figure'),
    [Input(component_id='frame-slider', component_property='value'),
     Input(component_id='snapshot-slider', component_property='value')]
)
def update_actions(frame, snapshot):
    logits = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0/logits'].value
    softmax_logits = F.softmax(torch.from_numpy(logits), dim=1).numpy()
    traces = []
    actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    for a in range(softmax_logits.shape[1]):
        trace = dict(
        x = list(range(0, softmax_logits.shape[0], 5)),
        y = softmax_logits[::5, a],
        hoverinfo = 'x+y',
        mode = 'lines',
        line = dict(width=0.5),
        stackgroup = 'one',
        name = actions[a]
    )
        traces.append(trace) 
    xaxis=dict(
        title='frame')
    yaxis=dict(
        title='Softmax value')
    figure = go.Figure(data = traces, layout=go.Layout(xaxis=xaxis, yaxis=yaxis))
    return figure

# @app.callback(
#     Output(component_id='actions', component_property='figure'),
#     [Input(component_id='snapshot-slider', component_property='value')]
# )
# def update_epr(snapshot):

#     return figure

def saliency_on_frame_abbr(S, frame, fudge_factor, sigma = 0, channel = 0):
    S = fudge_factor * S / S.max()
    I = frame.astype('uint16')
    I[35:195,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

@app.callback(
    Output(component_id='screen-ins', component_property='src'),
    [Input(component_id='frame-slider', component_property='value'),
     Input('snapshot-slider', 'value')]
)
def update_frame_in_slider(frame, snapshot):
    # fetch frame based on snapshot and frame
    ins = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0/ins'].value
    img = ins.copy()
    history = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
    actor_frames = history['actor_sal'].value
    critic_frames = history['critic_sal'].value
    if frame > len(img):
        img = np.zeros((210,160,3))
        actor_frames = img.copy(); critic_frames = img.copy()
    else: 
        img = img[frame]
        actor = actor_frames[int(frame/100)]; critic=critic_frames[int(frame/100)]
    
        
    img = saliency_on_frame_abbr(actor, img, 300, 0, 2)
    img = saliency_on_frame_abbr(critic, img, 400, 0 , 0)
    
    buffer = BytesIO()
    plt.imsave(buffer, img)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return 'data:image/png;base64,{}'.format(img_str)

@app.callback(
    Output(component_id='regions_subplots', component_property='figure'),
    [Input(component_id='frame-slider', component_property='value'),
     Input(component_id='snapshot-slider', component_property='value')]
)
def update_regions_plots(frame, snapshot):
    ins = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0/ins'].value
    img = ins.copy()
    if frame > len(img):
        img = np.zeros((210,160,3))
    else: img = img[frame]
    
    ymid, xmid = 110, 80
    
    it = 30

    history = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
    actor_frames = history['actor_sal'].value
    critic_frames = history['critic_sal'].value
    
    actor_tot = actor_frames.sum((1,2))
    critic_tot = critic_frames.sum((1,2))
   
    targets = [(actor_frames[:, :40, :40], critic_frames[:, :40, :40]),
               (actor_frames[:, :40, 40:], critic_frames[:, :40, 40:]),
               (actor_frames[:, 40:, :40], critic_frames[:, 40:, :40]),
               (actor_frames[:, 40:, 40:], critic_frames[:, 40:, 40:])]
    # intensity defined by sum of values in frame region divided by sum of total values of full frame
    
    trace_labels = ['TopLeft', 'TopRight', 'BotLeft', 'BotRight']
    
    a_traces = []
    for i in range(4):
        trace = dict(
            x = list(range(0, actor_frames.shape[0] * 100, 100)),
            y = (targets[i][0]).sum((1,2)) / actor_tot,
            hoverinfo = 'x+y',
            line = dict(
                color = ('rgb(24, 12, 205)'),
                width = 3)
        )

        a_traces.append(trace)
        
    c_traces = []
    for i in range(4):
        trace = dict(
            x = list(range(0, actor_frames.shape[0] * 100, 100)),
            y = (targets[i][1]).sum((1,2)) / critic_tot,
            hoverinfo = 'x+y',
            line = dict(
                color = ('rgb(205, 12, 24)'),
                width = 3)
        )

        c_traces.append(trace)
    fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Top left', 'Top Right',
                                                          'Bottom left', 'Bottom Right'))
    
    for series in [a_traces, c_traces]:
        fig.append_trace(series[0], 1, 1)
        fig.append_trace(series[1], 1, 2)
        fig.append_trace(series[2], 2, 1)
        fig.append_trace(series[3], 2, 2)
    
    fig['layout'].update(title='Saliency intensity by quarter region', showlegend=False)
    fig['layout']['xaxis3'].update(title='Frame')
    fig['layout']['xaxis4'].update(title='Frame')
    fig['layout']['yaxis1'].update(title='Intensity', range=[0,1])
    fig['layout']['yaxis3'].update(title='Intensity', range=[0,1])
    fig['layout']['yaxis2'].update(range=[0,1])
    fig['layout']['yaxis4'].update( range=[0,1])
    fig['layout']['xaxis1'].update(anchor='x3')
    fig['layout']['xaxis2'].update(anchor='x4')

    return fig


# def update_frame_in_slider(frame, snapshot):
#     # fetch frame based on snapshot and frame
#     images = 'static/images'
#     avail = os.listdir(images)
#     if str(snapshot) not in avail:
#         return os.path.join(images, 'dead.png') # some default val, return something else later
    
#     snapshot_dir = os.path.join(images, str(snapshot))
#     if str(frame) not in [name.split('.')[0] for name in os.listdir(snapshot_dir)]:
#         return os.path.join(images, 'dead.png') # if frame not there
#     #print(os.path.join(snapshot_dir, str(frame)+'.png'))
#     return os.path.join(snapshot_dir, str(frame)+'.png')
    

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

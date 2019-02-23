# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
import h5py
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

log_data = pd.read_csv("baby-a3c/breakout-v4/log-model7-02-17-20-41.txt")
log_data.columns = log_data.columns.str.replace(" ", "")

app.layout = html.Div(children=[
    html.H1(children='Interactive Atari RL'),
    html.Div([
        html.Div([
            html.Div(id='frame-val'),
            dcc.Slider(id='frame-slider',
                   min = 0,
                   max = 3000,
                   value = 0,
                   marks = {i: str(i) for i in range(0, 3000, 200)},
                   step = None
               
                  )
        ], style = {'padding-bottom':'50px'}),
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
    html.Div(id = 'screen-ins'),
    html.Div(children=[dcc.Graph(
                           id='mean-epr_over_eps',
                           figure={
                               'data': [
                                   #py.iplot(log_data['episodes'], log_data['mean-epr'])
                                   go.Scatter(x=log_data['episodes'],
                                              y=log_data['mean-epr'])
                               ],
                               'layout': {
                                   'xaxis': {'title': 'Episodes'},
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
                                   go.Scatter(x=log_data['episodes'],
                                              y=log_data['run-loss'])

                               ],
                               'layout': {
                                   'xaxis': {'title': 'Episodes'},
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
    Output(component_id='screen-ins', component_property='children'),
    [Input(component_id='frame-slider', component_property='value'),
     Input('snapshot-slider', 'value')]
)
def update_frame_in_slider(frame, snapshot):
    # fetch frame based on snapshot and frame
    store = h5py.File('visualize_atari/model_rollouts.h5', 'r')
    dest = list(store.keys())[0] # just take first find here- clean later 
    dest += '/model.' + str(snapshot) + '.tar/history/ins'
    if frame >= len(store[dest]):
        target = np.array(shape=(210, 160, 3))
    else:
        target = store[dest][frame]
    store.close()
    return 'target shape: {}'.format(dest)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

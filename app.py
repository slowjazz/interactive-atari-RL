# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import matplotlib.pyplot as plt
import PIL
import h5py
import numpy as np
import os, base64
from io import BytesIO
from scipy.stats import entropy

import gym
from visualize_atari import *

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

# Total training log
log_data = pd.read_csv("baby-a3c/breakout-v4/log-model7-02-17-20-41.txt")
log_data.columns = log_data.columns.str.replace(" ", "")


# Load data into memory by default, otherwise delete 'driver='core'' to read from disk
replays = h5py.File('static/model_rollouts_5.h5','r', driver='core')

# list of epoch numbers we took (number * 500k is the number of frames trained at that point)
snapshots = [1,19,30,40,50,60,70,80,90,100]

# HTML Page layout
app.layout = html.Div(children=[
    html.H5(children='Interactive Atari RL', id = 'null',style = {'padding-bottom':'0px', 'margin':'0'}),
    # Top Row
    html.Div([
        html.Div([
            dcc.Graph(id = 'rewards-candlestick',
                      style={'height':'21em'}
                     )
        ], style={'border':'1px solid black', 'display':'inline-block'}),
        html.Div([
            dcc.Graph(id = 'all-cum-rewards',
                      style={'height':'21em'}
                     )
        ], style={'border':'1px solid black', 'display':'inline-block'}),
        
    ], style = {'height':'23em','overflow':'auto','display':'block','width':'150em'}),
    
    html.Div([ # Big bottom group with 3 columns
        html.Div([ # column 1
          
            html.Div([

                  html.Div(style = {'padding-top':'0em',
                                    'padding-left':'0',
                                    },
                           id = 'info-box-epoch'
                          ),
                  html.Div(style = {'padding-top':'0em',
                                    'padding-left':'0',
                                    },
                           id = 'info-box-frame'
                          )
                 ],
                 style={'border':'1px solid black', 'display':'inline-block'}),
            
            html.Div([
                html.Button('5 frames back', id='back-frame', style={'display':'inline-block','width':'50%'}),
                html.Button('5 frames next', id='forward-frame', style={'display':'inline-block','width':'50%'}),
            ],style={'border':'1px solid black', 'display':'block'}), 
            
            html.Div(html.Img(id = 'screen-ins',
                                   style = {'max-width':'100%', 'max-height':'100%','height':'30em'}),
                           style = {'border':'1px solid black','width':'20em','height':'100%'}),
        ], style = {'position':'absolute','top':'8px','border':'1px solid black', 'display':'inline-block','width':'20em','border-bottom':'20em'}),
        html.Div([ # column 2a (gantt and parallel coords plots); (data timeline plots referred to 'gantt' in code because it was my original name for these plots as I thought they looked like gantt charts)
            html.Div([
                dcc.Graph(id = 'gantt',
                          style={'border':'1px solid black','height':'15em','display':'block'})
                    ], style = {'width':'60em'}),
             html.Div([
                dcc.Graph(id = 'gantt2',
                          style={'border':'1px solid black','height':'15em','display':'block'})
                    ], style = {'width':'60em'}),
            html.Div([
                dcc.Graph(id = 'parallel-sal',
                          style={'border':'1px solid black','height':'30em','display':'block'})
                    ], style = {'width':'60em'}),
            
        ], style = {'position':'absolute','margin-left':'20em','border':'1px solid black', 'display':'inline-block'}),
        html.Div([ # column 2b (dropdowns)
            html.Div(["Select episodes for 'gantt' and parallel coordinates plots"], style = {'word-wrap':'break-word'}),
            html.Div([
                dcc.Dropdown(
                    id='gantt-select1',
                    options=[{'label':x, 'value':x} for x in snapshots],
                    value='90'
                ),
                dcc.Dropdown(
                    id='gantt-select2',
                    options=[{'label':x, 'value':x} for x in snapshots],
                    value='60'
                ),], style = {'padding-bottom':'15em'}),
            
        ], style = {'position':'absolute','margin-left':'80em','width':'6em','border':'1px solid black', 'display':'inline-block'}),
        
        html.Div([ # column 3 (saliency by region heatmap and subplots)
            html.Div([
                dcc.Graph(id = 'regions_bars',
                         style = {'border':'1px solid black', 'height':'23em'})
            ]),
            
            html.Div([
                dcc.Graph(id = 'regions-subplots',
                         style = {'height':'37em'})
            ], style = {'display':'block','border':'1px solid black'})
            
        ], style = {'position':'absolute','border':'1px solid black', 'display':'inline-block','margin-left':'86em'}),
        
        html.Div([ # column 4 (action and trajectory plots)
            html.Div([
                dcc.Graph(id = 'actions',
                          style={'border':'1px solid black','height':'23em'})
                    ], ),
            html.Div([
                dcc.Graph(id = 'trajectory',
                         style = {'border':'1px solid black'})
            ])
            
        ], style = {'position':'absolute','margin-left':'135em','border':'1px solid black', 'display':'inline-block'}),
    ], style = {'position':'relative', 'display':'block'}),

    html.Div([ # Row 3 (sliders)
        html.Div([
            html.Div(id='frame-val'),
            dcc.Slider(id='frame-slider',
                   min = 0,
                   max = 3000,
                   value = 0,
                   marks = {i: str(i) for i in range(0, 3000, 100)},
                   step = 5
               
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
    ], style={'position':'relative','padding-bottom':'20px','margin-top':'65em'}),
    html.Div([ # Row 4 (Action entropy (sort of legacy))
        dcc.Graph(id = 'action-entropy')
    ], style={'border':'1px solid black'}),
    html.Div(50, id = 'current-frame'), # 'hidden' div to store current frame as state
    html.Div(0, id = 'gantt-mem') # 'hidden' div to keep track of last frame called from first gantt (needed since second gantt plot couldn't update frame otherwise)
           
])

#====================== Begin callback functions, serving as the 'backend' to control all plotting =================#
# Each callback function is decorated with the named HTML components used for inputs and outputs
# https://dash.plot.ly/getting-started-part-2

# Update written values on page with the frame slider
@app.callback(
    [Output(component_id='frame-val', component_property='children'),
     Output(component_id='current-frame', component_property='children'),
     Output(component_id='gantt-mem', component_property='children')],
    [Input(component_id='frame-slider', component_property='value')]
)
def update_frame_slider(input_value):
    return 'Frame number of episode: {}'.format(input_value), input_value, input_value

# Update written values on page with the snapshot slider and mark unavailable values as red on the frame slider
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

# Update info box above frame display with current snapshot val
@app.callback(
    Output(component_id='info-box-epoch', component_property='children'),
    [Input(component_id='snapshot-slider', component_property='value')]
)
def update_info_box(input_value):
    return f'Epoch selected: {input_value}'

# Update info box above frame display with current frame val 
@app.callback(
    Output(component_id='info-box-frame', component_property='children'),
    [Input(component_id='frame-slider', component_property='value')]
)
def update_info_box_frame(input_value):
    return f'Frame selected: {input_value}'

# Handle all links controlling snapshots by passing to the snapshot slider
@app.callback(
    Output(component_id='snapshot-slider', component_property='value'),
    [Input(component_id='action-entropy', component_property = 'clickData'),
     Input('rewards-candlestick', 'clickData'),
     Input(component_id='gantt', component_property='clickData'),
     Input(component_id='gantt2', component_property='clickData')],
    [ State('gantt-mem', 'children'),
        State(component_id='gantt-select1', component_property='value'),
     State(component_id='gantt-select2', component_property='value')]
)
def update_link_snapshot(entropy_click, candle_click, gantt_click1, gantt_click2, gantt_click_memory, gantt_epoch1, gantt_epoch2):    
    if entropy_click:
        return (entropy_click['points'][0]['x'])
    if candle_click:
        return (candle_click['points'][0]['x'])
    print(gantt_click1, gantt_click_memory)
    # Prevent first gantt's clicks from re-updating if it didn't change    
    if gantt_click1 and myround(gantt_click1['points'][0]['x']) != gantt_click_memory:
        print(gantt_epoch1)
        return gantt_epoch1
    if gantt_click2:
        print(gantt_epoch2)
        return gantt_epoch2
    
    return 50

# Round to multiple of 5
def myround(x, base=5):
    return base * round(x/base)

# Handle all links controlling frame by passing to the frame slider
@app.callback(
    Output(component_id='frame-slider', component_property='value'),
    
    [Input(component_id='regions-subplots', component_property = 'clickData'), 
     Input('actions', 'clickData'),
     Input('trajectory', 'clickData'),
     Input('regions_bars', 'clickData'),
     Input('gantt', 'clickData'),
     Input('gantt2', 'clickData'),
    Input(component_id='back-frame', component_property='n_clicks'),
    Input(component_id='forward-frame', component_property='n_clicks'),
    ],
    [State(component_id='current-frame', component_property='children'),
     State('gantt-mem', 'children'),],
)
def update_link_frame(regions_click, actions_click, trajectory_click, bars_click, 
                      gantt_click1,  gantt_click2,
                      back_click, forward_click,
                      cur_frame, gantt_click_memory):
    ctx = dash.callback_context
    # Check which buttons was pressed (since we only have n_clicks data)
    for item in ctx.triggered:
        if 'back-frame' in item['prop_id'] and item['value']:
            return max(0, cur_frame - 5)
        if 'forward-frame' in item['prop_id'] and item['value']:
            return cur_frame +5

    # Else, check through inputs of specified plots
    if regions_click:
        return (regions_click['points'][0]['x'])
    if actions_click:
        return myround(actions_click['points'][0]['x'])
    if trajectory_click:
        return myround(trajectory_click['points'][0]['x'])
    if bars_click:
        return (bars_click['points'][0]['x'])
     # While not buttons, clicking on lines in gantt plots need to be put here for some response errors
    if gantt_click1 and myround(gantt_click1['points'][0]['x']) != gantt_click_memory:
        return myround((gantt_click1['points'][0]['x']))
    if gantt_click2:
        print(gantt_click2)
        return myround((gantt_click2['points'][0]['x']))

    return 50

# Control rewards candlestick chart (row 1, col 1)
@app.callback(
    Output(component_id='rewards-candlestick', component_property='figure'),
    [Input(component_id='null', component_property='children')]
)
def update_rewards_candlestick(start):
    # Get data in increments of 40 (less noise)
    epr_xrange = (log_data['frames']/500e3).values[::40]
    epr_vals = log_data['mean-epr'].values[::40]
    rewards_candle_hovertext = [str(i) for i in epr_vals[:-1]]
    # Derive plot with data
    trace = go.Ohlc(x = epr_xrange,
                    open = epr_vals[:-1],
                    high = epr_vals[:-1],
                    low = epr_vals[1:],
                    close = epr_vals[1:],
                    text = rewards_candle_hovertext,
                    hoverinfo = 'x+text',
                    name = 'Mean EPR')
    data = [trace]
    
    # Also accumulate total saliency per episode for both types
    saliency_toplevel = []
    for s in snapshots:
        print(s)
        history = replays['models_model7-02-17-20-41/model.'+str(s)+'.tar/history/0']
        actor_frames, critic_frames = history['actor_sal'].value, history['critic_sal'].value
        actor_tot_perframe = actor_frames.sum((1,2)).sum()/actor_frames.shape[0]

        critic_tot_perframe = critic_frames.sum((1,2)).sum()/critic_frames.shape[0]
        
        saliency_toplevel.append([actor_tot_perframe, critic_tot_perframe])
    
    saliency_toplevel = np.array(saliency_toplevel)
    
    # Write saliency data into bars
    actor_bars = go.Bar(
        x = snapshots,
        y = saliency_toplevel[:,0],
        name = 'A Sal /frame',
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(68, 68, 230, 0.7)'
        ),
        hoverinfo = 'none'
    )
    critic_bars = go.Bar(
        x = snapshots,
        y = saliency_toplevel[:,1],
        name = 'C Sal /frame',
        yaxis='y2',
        marker = dict(
            color = 'rgba(240, 68, 68, 0.7)'
        ),
        hoverinfo = 'none'
    )
    
    data += [actor_bars, critic_bars]
    
    
    layout = go.Layout(
                 title = "Mean episode reward",
                 xaxis = dict(
                     title = "Frames (500k)",
                     rangeslider = dict(
                         visible = False
                     )
                 ),
                 yaxis = dict(
                     title = "Reward"
                 ),
                 yaxis2=dict(
                     title='Total saliency per frame',
                     overlaying='y',
                     side='right',
                     range = [0, 1200e3],
                     rangemode = 'nonnegative'
                 ),
                 margin = dict(
                     l = 50,
                     r = 60,
                     b = 35,
                     t = 30,
                     pad = 4
                 ),
                 legend = dict(x = 0.2, y = 1)
             )
    
    figure = go.Figure(data = data, layout = layout)
    return figure

# Control cumulative rewards plot (row 1, col 2)
# Input is 'null' as it stays static 
@app.callback(
    Output(component_id='all-cum-rewards', component_property='figure'),
    [Input(component_id='null', component_property='children')]
)
def update_all_cum_rewards(null):
    data = []
    for s in snapshots: 
        rewards = replays['models_model7-02-17-20-41/model.'+str(s)+'.tar/history/0/reward'].value
        reward_trace = dict(
            y = np.cumsum(rewards),
            x = list(range(0, rewards.shape[0])),
            name = f'Epoch {s}',
            line = dict(width = 3)
        )
        data.append(reward_trace)
    
    layout = go.Layout(title = 'Cumulative reward by epoch replay',
                       xaxis=dict(title='Episode Frame'), 
                       yaxis=dict(title='Reward'),
                       margin = dict(
                         l = 50,
                         r = 40,
                         b = 35,
                         t = 30,
                         pad = 4
                       ),
                       #legend = dict(x = 0.1, y = 1),
                       showlegend = False
                       )
    figure = go.Figure(data = data, layout= layout)
    return figure

# Control action entropy graph at bottom (row 4, col 1)
@app.callback(
    Output(component_id='action-entropy', component_property='figure'),
    [Input(component_id='null', component_property='children')]
)
def update_actions_entropy(null):
    return go.Figure() # turn off 
    # Get list of available snapshots
    iterations = sorted([int(x.split('.')[1]) for x in list(replays['models_model7-02-17-20-41'].keys())])
    y_data = []
    ep_lengths = {}
    x_range = []
    y_segments = {i:[] for i in range(4)}
    avg_len = 20
    actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    # For each iteration, get logits and scale on x-axis with respect to length of episode
    # Logits are converted to moving averages of window size 10
    for i in iterations:
        softmax_logits = replays['models_model7-02-17-20-41/model.'+str(i)+'.tar/history/0/outs'].value
        y_data.append(softmax_logits)
        ep_lengths[i] = len(softmax_logits)
        x_range.append(10*np.arange(0, (len(softmax_logits)/avg_len)))
        ids = np.arange(len(softmax_logits))//avg_len
        for a in range(4):
            y_segments[a].append(np.bincount(ids,softmax_logits[:,a])/np.bincount(ids))

    # Also get entropy data - entropy is calculated by action for the entire episode
    entropy_data = np.array([entropy(logits) for logits in y_data])
    softmax_data = np.vstack(y_data)
    
    # Put stacked logits segments together
    x_range = np.hstack(x_range)
    for a in range(4):
        y_segments[a] = np.hstack(y_segments[a])
    
    # Create plot
    ids = np.arange(len(softmax_data))//avg_len
    series = [entropy_data[:, i] for i in range(4)] 
    data = []
    # Iterate by action type
    for i, t in enumerate(series):
        # plot entropy as lines
        trace = go.Scatter(
                y = t,
                x = iterations,
                name = actions[i]
        )
        data.append(trace)
        
        # Plot moving averaged logits as stacked lines
        averaged = np.bincount(ids,softmax_data[:,i])/np.bincount(ids)
        trace2 = dict(
            x = 101*np.arange(0, (len(softmax_data)/avg_len), 1/(len(softmax_data)/avg_len)),
            y = averaged,
            mode = 'lines',
            line = dict(width=0.5),
            stackgroup = 'one',
            yaxis='y2',
            name = actions[i]
        )
        data.append(trace2)
    
    
    
    layout = go.Layout(title = 'Entropy by Action per Iteration Episode', 
                       height = 300,
                       yaxis=dict(
                           title='Entropy'
                       ),
                       yaxis2=dict(
                           title='Softmax',
                           overlaying='y',
                           side='right'
                       ),
                       clickmode = 'event+select')
    figure = go.Figure(data = data, layout = layout)
    return figure
    
# Control actions plot of stacked logits and cumulative reward per episode (row 1, column 4)
@app.callback(
    Output(component_id='actions', component_property='figure'),
    [Input(component_id='snapshot-slider', component_property='value')]
)
def update_actions(snapshot):
    softmax_logits = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0/outs'].value
    traces = []
    actions = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    # Just plot each logit for each frame
    for a in range(softmax_logits.shape[1]):
        trace = dict(
        x = list(range(0, softmax_logits.shape[0])),
        y = softmax_logits[:, a],
        hoverinfo = 'x+y',
        mode = 'lines',
        line = dict(width=0.5),
        stackgroup = 'one',
        name = actions[a]
    )
        traces.append(trace) 
    
    # Also plot cumulative rewards by frame
    rewards = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0/reward'].value
    reward_trace = dict(
        y = np.cumsum(rewards),
        x = list(range(0, softmax_logits.shape[0])),
        name = 'Ep Reward',
        yaxis="y2",
        line = dict(width = 3)
    )
    traces.append(reward_trace)
    layout = go.Layout(xaxis=dict(title='frame'), 
                       yaxis=dict(title='Softmax value'),
                       yaxis2=dict(
                           title='Rewards',
                           overlaying='y',
                           side='right'
                       ),
                      margin = dict(
                         l = 50,
                         r = 40,
                         b = 35,
                         t = 30,
                         pad = 4
                       ),
                       legend = dict(
                           orientation = "h",
                           y = 1.13)
                      )
    figure = go.Figure(data = traces, layout= layout)
    return figure

# Function from Greydanus to upscale saliency values into visible blots of blue/red
def saliency_on_frame_abbr(S, frame, fudge_factor, sigma = 0, channel = 0):
    S = fudge_factor * S / S.max()
    I = frame.astype('uint16')
    I[35:195,:,channel] += S.astype('uint16')
    I = I.clip(1,255).astype('uint8')
    return I

# Control frame display (row 2, col 1)
@app.callback(
    Output(component_id='screen-ins', component_property='src'),
    [Input(component_id='current-frame', component_property='children'),
     Input('snapshot-slider', 'value')]
)
def update_frame_in_slider(frame, snapshot):
    print('frame update')
    # fetch frame based on snapshot and frame
    frame = int(frame/5)
    ins = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0/ins'].value
    img = ins.copy()
    history = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
    # Get appropriate saliency values
    if frame > len(ins):
        img = np.zeros((210,160,3))
        actor = img.copy(); critic = img.copy()
    else: 
        img = img[frame]
        actor_frames = history['actor_sal'].value
        critic_frames = history['critic_sal'].value
        actor = actor_frames[frame]; critic=critic_frames[frame]
    # Overlay saliency on frame
    img = saliency_on_frame_abbr(actor, img, 500, 0, 2)
    img = saliency_on_frame_abbr(critic, img, 500, 0 , 0)
    
    # Save as base64 string
    buffer = BytesIO()
    plt.imsave(buffer, img)
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return 'data:image/png;base64,{}'.format(img_str)

# Helper func to fetch appropriate layout parameters for color, size, and border for an array of actions (0,1,2,3,2 etc) for 'gantt' plots
def actions_to_marker(actions):
    action_colors = {0: "rgb(87, 137, 224)", 1:'rgb(247, 227, 116)', 2:'rgb(59, 229, 73)', 3:'rgb(232, 73, 64)'}
    colors = [action_colors[i] for i in actions]
    return dict(color = colors, size = 8, line = dict(width = 1))

# Big function to generate 'gantt' plots (row 2, col 2)
# Give a snapshot number and object for selected data in the parallel coords plot below
# also take last arg to correct for zooming scale
def gantt_figures(snapshot, parallelSelectedData, range_bounds=None):
    ymid, xmid = 80, 80 # size in pixels of midpoints of saliency frames
    sal_thresh = 0.5 # threshold for selecting saliency as percentage of max saliency in frame
    action_thresh = 0.7 # threshold for selecting actions with softmax val greater than threshold
    
    # Generate saliency boxes 
    # This version selects frames where the *regional* saliency is greater than some threshold of the max *regional* saliency for the episode, and values are max-normalized by region
    def chart_data(snapshot):
        # Get data
        history1 = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
        rewards1 = history1['reward'].value
        outs = history1['outs'].value
        critic_sal = history1['critic_sal'].value
        lower, upper = 0, len(rewards1)

        # Subset data if there are range bounds (due to zooming)
        if range_bounds:
            lower = int(range_bounds[0])
            upper = int(range_bounds[1])
            rewards1 = rewards1[lower:upper]
            outs = outs[lower:upper]
            critic_sal = critic_sal[lower//5:upper//5]
        
        # subset actions that meet threshold 
        actions1ix = np.where(np.max(outs, axis = 1) > action_thresh)
        actions1types = np.argmax(outs[actions1ix], axis=1)

        # divide saliency into regions
        csaliency1 = critic_sal
        csaliency1regions = np.array([csaliency1[:, :ymid, :xmid], csaliency1[:, :ymid, xmid:], csaliency1[:, ymid:, :xmid], csaliency1[:, ymid:, xmid:]])

        csaliency1regions_maxs = [region.sum((1,2)).max() for region in csaliency1regions]

        # select frames where at least one of the saliency regions meet threshold for that region
        csaliency1ix = np.where((csaliency1regions[0].sum((1,2)) > sal_thresh*csaliency1regions_maxs[0]) | 
                                (csaliency1regions[1].sum((1,2)) > sal_thresh*csaliency1regions_maxs[1]) |
                                (csaliency1regions[2].sum((1,2)) > sal_thresh*csaliency1regions_maxs[2]) |
                                (csaliency1regions[3].sum((1,2)) > sal_thresh*csaliency1regions_maxs[3]))
        
        # Aggregate and reshape
        csaliency1frames = csaliency1regions[:, csaliency1ix]
        csaliency1frames = csaliency1frames.sum((3,4))
        csaliency1frames = np.squeeze(csaliency1frames)
        csaliency1frames = np.swapaxes(csaliency1frames, 0, 1)
        
        # Normalize by region max
        for region_ix in range(csaliency1frames.shape[1]):
            csaliency1frames[:, region_ix] /= csaliency1regions_maxs[region_ix]
        
        # At this point, csaliency1ix is an array of frame numbers and csaliency1frames is accumulated saliency values by region
        # The latter has shape N x 4, where N is the number of frames found that meet threshold and 4 is the number of regions
      


    ##### This version selects frames where the *total* saliency is greater than some threshold of the max *total* saliency for the episode

    # def chart_data(snapshot):
    #     history1 = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
    #     rewards1 = history1['reward'].value

    #     actions1ix = np.where(np.max(history1['outs'].value, axis = 1) > 0.7)
    #     actions1types = np.argmax(history1['outs'].value[actions1ix], axis=1)

    #     csaliency1 = history1['critic_sal'].value
    #     csaliency1sums = csaliency1.sum((1,2))
    #     csaliency1max = csaliency1sums.max()
        
    #     csaliency1ix = np.where(csaliency1sums > 0.3*csaliency1max)
    #     csaliency1frames = csaliency1[csaliency1ix]
        
    #     csaliency1regions = np.array([[x[:ymid, :xmid].sum(), x[:ymid, xmid:].sum(), x[ymid:, :xmid].sum(), x[ymid:, xmid:].sum()] for x in csaliency1frames])
        
    #     csaliency1regions /= csaliency1max
    
        # convert n x n_regions array of saliency values to traces that look like the 2x2 saliency grids along a time line
        # example: we pass in n x [1,2,3,4] as values, where 1,2,3,4 are total saliency values of top left, top right, bot left, bot right regions
        def plot_region_dots(region_vals, region_ix):
            xvals, selPoints = [], []
            yvals = np.tile(np.array([1,1,0,0]), len(region_vals))
            opacities = ['rgba(200, 68, 68,'+ str(i) + ')' for i in region_vals.flatten()]
            width = int(0.024 * len(rewards1)) # scale saliency marks to what user currently sees (to accoutn for zooming)
            
            for i, ix in enumerate(region_ix[0]):
                xvals += [ix*5, ix*5 + width, ix*5, ix*5 + width]

                # Select points if they were highlighted in frame range specified by dimension 0 of parallel coords plot
                # Needed to parse through format of the event callback (was absolutely monstrous)
                if parallelSelectedData:
                    for dim in parallelSelectedData:
                        if type(dim)==dict:
                            for k, constraint_objs in dim.items():
                                if k == 'dimensions[0].constraintrange': # only parse frame data
                                    if type(constraint_objs[0][0]) == list:
                                        constraint_objs = constraint_objs[0]
                                    for c in filter(lambda x: x != None, constraint_objs):
                                        lower = c[0]
                                        upper = c[1]
                                        
                                        if ix >= lower//5 and ix <= upper//5:
                                            selPoints += [4*i, 4*i+1, 4*i+2, 4*i+3] # add indices of selected data
            # Return data with some styling
            return xvals, yvals, dict(color= opacities, size = 14, line = dict(width = 1), symbol = 'square'), selPoints
    
        
        # Collect data from above function 
        t1infox, t1infoy, markers, selectedpointsInfo = plot_region_dots(csaliency1frames, csaliency1ix)

        # Return all data for 'gantt' plots: rewards, actions, and saliency boxes
        return (np.array(t1infox) + lower, t1infoy, markers, selectedpointsInfo), (list(range(lower, upper)), rewards1), (actions1ix[0] + lower, rewards1[actions1ix], actions_to_marker(actions1types))

    # Get saliency, rewards, and actions data formatted to put into a chart
    trace1sal, trace1rewards, trace1actions = chart_data(snapshot)

    # Chart all data

    if not trace1sal[3]: # if no selected data
        trace1info = go.Scatter(
            mode = 'markers',
            x = trace1sal[0],
            y = trace1sal[1],
            marker = trace1sal[2],
            cliponaxis= False,
        )

    else:
        trace1info = go.Scatter(
            mode = 'markers',
            x = trace1sal[0],
            y = trace1sal[1],
            selectedpoints = trace1sal[3],
            marker = trace1sal[2],
            cliponaxis= False,
            unselected = dict(
                marker = dict(color = 'rgba(68, 68, 68,0.1)'),
                
            ),
            
        )
    
    # Rewards
    trace1 = go.Scatter(
        x = trace1rewards[0],
        y = trace1rewards[1],
    )

    # Actions
    trace1actions = go.Scatter(
        mode = 'markers',
        x = trace1actions[0],
        y = trace1actions[1],
        opacity = 1,
        marker = trace1actions[2],
        cliponaxis= False
    )

    # Make subplots skeleton
    fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                              shared_xaxes = True, shared_yaxes = True)

    
    fig.append_trace(trace1info, 1, 1)
    fig.append_trace(trace1, 2, 1)
    fig.append_trace(trace1actions, 2, 1)    

    # update layouts to control scale, axis labels
    fig['layout'].update(title='Data timeline chart of epoch ep '+str(snapshot), 
                         showlegend=False,
                         clickmode = 'event+select',
                         margin = dict(
                             l = 50,
                             r = 40,
                             b = 35,
                             t = 25,
                             pad = 4
                           ))

    fig['layout']['yaxis1'].update(tickmode='linear',
                                    ticks='outside',
                                    tick0=0,
                                    dtick=2,
                                  showticklabels=False,
                                  range = [0,5],
                                  fixedrange = True)
    
    fig['layout']['yaxis2'].update(title='', range=[0, 7], autorange=False)

    return fig


# Control 'gantt' charts (row 2, col 2) 
@app.callback(
    [Output(component_id='gantt', component_property='figure'),
    Output(component_id='gantt2', component_property='figure'),
    ],
    [Input(component_id='gantt-select1', component_property='value'),
    Input(component_id='gantt-select2', component_property='value'),
    Input(component_id='parallel-sal', component_property='restyleData'),
    Input(component_id='gantt', component_property='relayoutData'),
    Input(component_id='gantt2', component_property='relayoutData'), # listen for parallel coords linking
    ]
)
def update_gantts(snapshot1, snapshot2, parallelSelectedData, relayout1, relayout2):
    if not snapshot1: # default values on page load
        snapshot1 = 90
    if not snapshot2:
        snapshot2 = 60

    range_bounds, range_bounds2 = None, None

    # If zoomed, receive a callback to properly scale the 2x2 saliency marks
    # fetch x-axis range bounds from each callback
    if relayout1 and 'xaxis.range[0]' in relayout1:
        range_bounds = [relayout1['xaxis.range[0]'], relayout1['xaxis.range[1]']]
    if relayout2 and 'xaxis.range[0]' in relayout2:
        range_bounds2 = [relayout2['xaxis.range[0]'], relayout2['xaxis.range[1]']]
    
    fig1 = gantt_figures(snapshot1, parallelSelectedData, range_bounds)
    fig2 = gantt_figures(snapshot2, parallelSelectedData, range_bounds2)
    return fig1, fig2


# Control parallel coords plot (row 2, col 2)
@app.callback(
    Output(component_id='parallel-sal', component_property='figure'),
    [Input(component_id='gantt-select1', component_property='value'),
    Input(component_id='gantt-select2', component_property='value')],
)
def update_parallel_sal(snapshot1, snapshot2):
    ymid, xmid = 80, 80 # pixel lengths to middle of frame
    sal_thresh = 0.5 # threshold for selecting saliency as percentage of max saliency in frame
    
    # This version selects frames where the *regional* saliency is greater than some threshold of the max *regional* saliency for the episode, and values are max-normalized by region
    # Same as chart_data in gantt functions
    def chart_data(snapshot):
        history1 = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
        rewards1 = history1['reward'].value

        csaliency1 = history1['critic_sal'].value
        csaliency1regions = np.array([csaliency1[:, :ymid, :xmid], csaliency1[:, :ymid, xmid:], csaliency1[:, ymid:, :xmid], csaliency1[:, ymid:, xmid:]])

        csaliency1regions_maxs = [region.sum((1,2)).max() for region in csaliency1regions]

        csaliency1ix = np.where((csaliency1regions[0].sum((1,2)) > sal_thresh*csaliency1regions_maxs[0]) | 
                                (csaliency1regions[1].sum((1,2)) > sal_thresh*csaliency1regions_maxs[1]) |
                                (csaliency1regions[2].sum((1,2)) > sal_thresh*csaliency1regions_maxs[2]) |
                                (csaliency1regions[3].sum((1,2)) > sal_thresh*csaliency1regions_maxs[3]))
        
        csaliency1frames = csaliency1regions[:, csaliency1ix]
        csaliency1frames = csaliency1frames.sum((3,4))
        csaliency1frames = np.squeeze(csaliency1frames)
        csaliency1frames = np.swapaxes(csaliency1frames, 0, 1)
        
        for region_ix in range(csaliency1frames.shape[1]):
            csaliency1frames[:, region_ix] /= csaliency1regions_maxs[region_ix]
            
        return csaliency1ix[0]*5, csaliency1frames
    
        
        # This version selects frames where the *total* saliency is greater than some threshold of the max *total* saliency for the episode
        
#     def chart_data(snapshot):
#         history1 = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
#         rewards1 = history1['reward'].value

#         csaliency1 = history1['critic_sal'].value
#         csaliency1sums = csaliency1.sum((1,2))
#         csaliency1max = csaliency1sums.max()
        
#         csaliency1ix = np.where(csaliency1sums > 0.4*csaliency1max)
#         csaliency1frames = csaliency1[csaliency1ix]
        
#         csaliency1regions = np.array([[x[:ymid, :xmid].sum(), x[:ymid, xmid:].sum(), x[ymid:, :xmid].sum(), x[ymid:, xmid:].sum()] for x in csaliency1frames])
        
#         csaliency1regions /= csaliency1max

#         return csaliency1ix[0]*5, csaliency1regions

    # Get data
    ep1ix, ep1vals = chart_data(snapshot1)
    ep2ix, ep2vals = chart_data(snapshot2)

    max_range = max(ep1ix.max(), ep2ix.max())

    # Reshape data
    ep1data = np.hstack((ep1ix.reshape(-1, 1), ep1vals, np.repeat([int(snapshot1)], ep1ix.shape[0]).reshape(-1,1)))
    ep2data = np.hstack((ep2ix.reshape(-1, 1), ep2vals, np.repeat([int(snapshot2)], ep2ix.shape[0]).reshape(-1,1)))    
    
    # Put all data into 1 matrix
    all_data = np.vstack((ep1data, ep2data))
    
    # Chart data - frame is first dimension and regions follow on next 4 dimensions. Last column in all_data is the epoch number, which determines the color
    trace = go.Parcoords(
        line = dict(color = all_data[:, -1],
                    showscale = True),
        dimensions = list([
            dict(range = [0, max_range],
                label = 'Frame', values = all_data[:, 0]),
            dict(range = [0,1],
                label = 'TopLeft C-Sal', values = all_data[:,1]),
            dict(range = [0,1],
                label = 'TopRight C-Sal', values = all_data[:,2]),
            dict(range = [0,1],
                label = 'BotLeft C-Sal', values = all_data[:,3]),
            dict(range = [0,1],
                label = 'BotRight C-Sal', values = all_data[:,4]),
            
        ])
    )

    data = [ trace]
    layout = go.Layout( title = "Frames with by-region saliency > "+str(sal_thresh) + " of max",
                       margin = dict(
                         l = 55,
                         r = 50,
                         b = 35,
                         t = 80,
                         pad = 4
                       ),
                       
                      )
    figure = go.Figure(data = data, layout = layout)

    return figure

    
# Control region subplots (row 2, col 3)
@app.callback(
    Output(component_id='regions-subplots', component_property='figure'),
    [Input(component_id='snapshot-slider', component_property='value')]
)
def update_regions_plots(snapshot):    
    ymid, xmid = 80, 80
    window_length = 10

    # Get data
    history = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
    actor_frames = history['actor_sal'].value
    critic_frames = history['critic_sal'].value
    
    # Sum every frame
    actor_tot = actor_frames.sum((1,2))
    critic_tot = critic_frames.sum((1,2))
   
    # Divide frames into regions
    targets = [(actor_frames[:, :ymid, :xmid], critic_frames[:, :ymid, :xmid]),
               (actor_frames[:, :ymid, xmid:], critic_frames[:, :ymid, xmid:]),
               (actor_frames[:, ymid:, :xmid], critic_frames[:, ymid:, :xmid]),
               (actor_frames[:, ymid:, xmid:], critic_frames[:, ymid:, xmid:])]
    # intensity defined by sum of values in frame region divided by sum of total values of full frame
    
    trace_labels = ['TopLeft', 'TopRight', 'BotLeft', 'BotRight']
    
    a_traces = []
    c_traces = []
    a_ubounds, a_lbounds, c_ubounds, c_lbounds = [],[],[],[]
    for i in range(4):
        # Change indexing since saliency is only recorded every 5 frames
        xrange = list(range(0, actor_frames.shape[0] * 5, 5))

        # Get moving averages and moving mins/maxs of window_length (default 10)
        data = pd.Series((targets[i][0]).sum((1,2))/actor_tot).rolling(window=window_length)
        mavg = data.mean()
        lowerbound = data.min()
        upperbound = data.max()

        ubound = dict(
            x = xrange,
            y = upperbound,
            hoverinfo = 'x+y',
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(68, 68, 200, 0.3)',
            fill='tonexty'
        )
        trace = dict(
            x = xrange,
            y = mavg,
            hoverinfo = 'x+y',
            mode='lines',
            fillcolor='rgba(68, 68, 200, 0.3)',
            fill='tonexty',
            line = dict(
                color = ('rgb(24, 100, 205)'),
                width = 2)
        )
        lbound = dict(
            x = xrange,
            y = lowerbound,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines')
        
        a_ubounds.append(ubound)
        a_lbounds.append(lbound)
        a_traces.append(trace)
        
    # Do same as above but fro critic saliency
    for i in range(4):

        data = pd.Series((targets[i][1]).sum((1,2))/critic_tot).rolling(window=window_length)
        mavg = data.mean()
        lowerbound = data.min()
        upperbound = data.max()
        ubound = dict(
            x = xrange,
            y = upperbound,
            hoverinfo = 'x+y',
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            fillcolor='rgba(200, 68, 68, 0.3)',
            fill='tonexty'
        )
        trace = dict(
            x = xrange,
            y = mavg,
            hoverinfo = 'x+y',
            mode='lines',
            fillcolor='rgba(200, 68, 68, 0.3)',
            fill='tonexty',
            line = dict(
                color = ('rgb(205, 50, 24)'),
                width = 2)
        )
        lbound = dict(
            x = xrange,
            y = lowerbound,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines')

        c_ubounds.append(ubound)
        c_lbounds.append(lbound)
        c_traces.append(trace)
    
    fig = tools.make_subplots(rows=2, cols=2, 
                             horizontal_spacing = 0.05,
                             vertical_spacing = 0.05,
#                              subplot_titles=('Top left', 
#                                              'Top Right',
#                                              'Bottom left', 
#                                              'Bottom Right'),
                             )
    
    for series in [ a_lbounds, a_traces, a_ubounds, c_lbounds, c_traces, c_ubounds]:
        fig.append_trace(series[0], 1, 1)
        fig.append_trace(series[1], 1, 2)
        fig.append_trace(series[2], 2, 1)
        fig.append_trace(series[3], 2, 2)
    
    fig['layout'].update(title='Moving average Saliency intensity by quarter region of epoch '+str(snapshot), 
                         showlegend=False,
                         clickmode = 'event+select',
                         margin = dict(
                             l = 50,
                             r = 40,
                             b = 35,
                             t = 34,
                             pad = 4
                           ))
    fig['layout']['xaxis3'].update(title='Frame')
    fig['layout']['xaxis4'].update(title='Frame')
    fig['layout']['yaxis1'].update(title='Intensity', range=[0,1])
    fig['layout']['yaxis3'].update(title='Intensity', range=[0,1])
    fig['layout']['yaxis2'].update(range=[0,1])
    fig['layout']['yaxis4'].update( range=[0,1])
    fig['layout']['xaxis1'].update(anchor='x3')
    fig['layout']['xaxis2'].update(anchor='x4')

    return fig

# Control saliency region heatmap (row 2 col 3)
@app.callback(
    Output(component_id='regions_bars', component_property='figure'),
    [Input(component_id='snapshot-slider', component_property='value')]
)
def update_regions_bars(snapshot):
    ymid, xmid = 80, 80

    # Get data in same manner as for the region subplots
    history = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
    actor_frames = history['actor_sal'].value
    critic_frames = history['critic_sal'].value
    
    actor_tot = actor_frames.sum((1,2))
    critic_tot = critic_frames.sum((1,2))
    
    targets = [(actor_frames[:, :ymid, :xmid], critic_frames[:, :ymid, :xmid]),
               (actor_frames[:, :ymid, xmid:], critic_frames[:, :ymid, xmid:]),
               (actor_frames[:, ymid:, :xmid], critic_frames[:, ymid:, :xmid]),
               (actor_frames[:, ymid:, xmid:], critic_frames[:, ymid:, xmid:])]
    # intensity defined by sum of values in frame region divided by sum of total values of full frame
    
    trace_labels = ['TopLeft', 'TopRight', 'BotLeft', 'BotRight']
    
    # color intensity calculated as fraction of saliency intensity for the whole frame
    z = []
    for i, label in enumerate(trace_labels):
        z.append((targets[i][0]).sum((1,2)) / actor_tot)
    
    heatmap = go.Heatmap(
        z = z,
        x = list(range(0, actor_frames.shape[0]*5, 5)),
        y = trace_labels,
        colorscale = 'Viridis',
        yaxis='y2')
    
    data = [heatmap]
    layout = go.Layout(title = 'Actor Saliency Heatmap by Region of Epoch '+str(snapshot),
                       margin = dict(
                         l = 65,
                         r = 60,
                         b = 35,
                         t = 35,
                         pad = 4
                       ),)
    fig = go.Figure(data = data, layout = layout)
    return fig

# Control 'trajectory' plot (row 2, col 4) **not shown for final project**
@app.callback(
    Output(component_id='trajectory', component_property='figure'),
    [Input(component_id='snapshot-slider', component_property='value')]
)
def update_trajectory(snapshot):
    history = replays['models_model7-02-17-20-41/model.'+str(snapshot)+'.tar/history/0']
    softmax_logits = history['outs'].value
    actions = np.argmax(softmax_logits, axis=1)
    positions = np.zeros(softmax_logits.shape[0]+1)
    # movements are in pixels, but approximations, because position doesn't seem fully deterministic
    for i in range(len(actions)):
        if actions[i]==2: # go right
            positions[i+1] = max(positions[i] - 8, -96)
        elif actions[i] ==3: # go left
            positions[i+1] = min(positions[i] + 8, 96)
        else:
            positions[i+1] = positions[i]
   
    
    trace = go.Scatter(
        x = list(range(positions.shape[0]))[::10],
        y = positions[::10],
    )
    actor_frames, critic_frames = history['actor_sal'].value, history['critic_sal'].value
    
    actor_tot = actor_frames.sum((1,2))

    critic_tot = critic_frames.sum((1,2))
    
    actor_sum, critic_sum = actor_tot.max() , critic_tot.max()

    actor_bars = go.Bar(
        x = list(range(0, actor_frames.shape[0]*5, 5)),
        y = actor_tot/actor_sum,
        name = 'A Sal',
        yaxis = 'y2',
        marker = dict(
            color = 'rgba(68, 68, 230, 0.7)'
        ),
        hoverinfo = 'none'
    )
    critic_bars = go.Bar(
        x = list(range(0, actor_frames.shape[0]*5, 5)),
        y = critic_tot/critic_sum,
        name = 'C Sal',
        yaxis='y2',
        marker = dict(
            color = 'rgba(240, 68, 68, 0.7)'
        ),
        hoverinfo = 'none'
    )
    
    data = [trace, actor_bars, critic_bars]
    layout = go.Layout(title = 'Paddle Position',
                       xaxis=dict(
                           title='Frame'
                       ),
                      yaxis2=dict(
                       title='Frame saliency',
                       overlaying='y',
                       side='right',
                         ),
                      margin = dict(
                         l = 55,
                         r = 50,
                         b = 35,
                         t = 35,
                         pad = 4
                       ),
                      showlegend=False)
    fig = go.Figure(data = data, layout = layout)
    return fig
    

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

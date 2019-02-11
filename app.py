# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import plotly.plotly as py

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

log_data = pd.read_csv("baby-a3c/breakout-v4/log-modelTest4-02-10-22-28.txt")
log_data.columns = log_data.columns.str.replace(" ", "")

app.layout = html.Div(children=[
    html.H1(children='demo'),

    html.Div(children=["Dash: A web application framework for Python",
                       dcc.Graph(
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

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')

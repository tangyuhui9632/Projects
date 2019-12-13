# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 20:09:45 2018

@author: Yuhui Tang
"""

#import libraries
import plotly
plotly.tools.set_credentials_file(username='tangyuhui9632', api_key='dD9jnw8Uv9Rc4oZZdEgn')
import plotly.plotly as py
import pandas as pd
import numpy as np
import plotly.graph_objs as go



def processing_data(movie):
    data = pd.read_csv('cleaned_data.csv')
    selected_movie = data[data.Movie_name == movie]
    
    return selected_movie
    
    

def line_chart(selected_movie):
    
    # set the data for traces
    x_axis = selected_movie.Track_name
    acousticness = selected_movie.acousticness
    danceability = selected_movie.danceability
    energy = selected_movie.energy
    instrumentalness = selected_movie.instrumentalness
    liveness = selected_movie.liveness
    speechiness = selected_movie.speechiness
    valence = selected_movie.valence
    
    # Create traces
    trace0 = go.Scatter(
        x = x_axis,
        y = acousticness,
        mode = 'lines+markers',
        name = 'acousticness'
    )
    trace1 = go.Scatter(
        x = x_axis,
        y = danceability,
        mode = 'lines+markers',
        name = 'danceability'
    )
    trace2 = go.Scatter(
        x = x_axis,
        y = energy,
        mode = 'lines+markers',
        name = 'energy'
    )
    trace3 = go.Scatter(
        x = x_axis,
        y = instrumentalness,
        mode = 'lines+markers',
        name = 'instrumentalness'
    )
    
    trace4 = go.Scatter(
        x = x_axis,
        y = liveness,
        mode = 'lines+markers',
        name = 'liveness'
    )
    trace5 = go.Scatter(
        x = x_axis,
        y = speechiness,
        mode = 'lines+markers',
        name = 'speechiness'
    )
    trace6 = go.Scatter(
        x = x_axis,
        y = valence,
        mode = 'lines+markers',
        name = 'valence'
    )
    data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]

    # Edit the layout
    layout = dict(title = 'Distribution of music features of a given movie',
                  xaxis = dict(title = 'music feature value'),
                  yaxis = dict(title = 'soundtrack'),
                  )
    
    fig = go.Figure(data=data,layout=layout)
    
    py.iplot(data, filename='movie-line-chart')
    

if __name__ == "__main__":
    selected_movie = processing_data('La La Land')
    line_chart(selected_movie)
    
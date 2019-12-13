# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:21:17 2018

@author: Yuhui Tang
"""

# Import libraries
import plotly
plotly.tools.set_credentials_file(username='tangyuhui9632', api_key='dD9jnw8Uv9Rc4oZZdEgn')
import plotly.plotly as py
import plotly.graph_objs as go

import pandas as pd
import numpy as np

def loadData():
    # Load the dataset
    data = pd.read_csv('cleaned_data.csv')
    # Bin the popularity into two groups: popular & unpopular
    popularity = data.popularity
    binnames = ['popular','unpopular']
    bins1=[-0.1,40,100] # extend the lower bound from 0 to -0.1 to include the minimum value of popularity
    popGroups = pd.cut(data['popularity'], bins1, labels=binnames)
    # Add the binned group into dataset
    data['PopGroups'] = popGroups
    return data

def acousticBoxplot(data):
    # get the acoustic value for popular and unpopular groups
    acousticPop = data.loc[data['PopGroups'] == 'popular']['acousticness'].values
    acousticUnpop = data.loc[data['PopGroups'] == 'unpopular']['acousticness'].values
    # draw the boxplot
    trace0 = go.Box(
        x=acousticPop,
        name = 'Popular',
        boxmean=True,
        marker = dict(
            color = 'rgba(255, 144, 14, 0.5)'),
        line = dict(
            color = 'rgba(255, 144, 14, 0.5)')
    
    )
    trace1 = go.Box(
        x=acousticUnpop,
        name = 'Unpopular',
        boxmean=True,
        marker = dict(
            color = 'rgba(207, 114, 255, 0.5)'),
        line = dict(
            color = 'rgba(207, 114, 255, 0.5)')
    )
    
    graph = [trace0, trace1]
    layout = {
        'xaxis': {
            'title': 'Acousticness',
            'zeroline': False,
        }
    }
    
    fig = go.Figure(data=graph,layout=layout)
    py.iplot(fig, filename = "Box Plot on Acousticness vs.Popularity")



def instruBoxplot(data):
    # get the instrumentalness value for popular and unpopular groups
    instruPop = data.loc[data['PopGroups'] == 'popular']['instrumentalness'].values
    instruUnpop = data.loc[data['PopGroups'] == 'unpopular']['instrumentalness'].values
    # plot the boxplot
    trace0 = go.Box(
        x=instruPop,
        name = 'Popular',
        boxmean=True,
        marker = dict(
            color = 'rgba(93, 164, 214, 0.5)'),
        line = dict(
            color = 'rgba(93, 164, 214, 0.5)'),
        boxpoints='all'
    )
    
    trace1 = go.Box(
        x=instruUnpop,
        name = 'Unpopular',
        boxmean=True,
        marker = dict(
            color = 'rgba(44, 160, 101, 0.5)'),
        line = dict(
            color = 'rgba(44, 160, 101, 0.5)'),
        boxpoints='all'
    )
    
    graph = [trace0, trace1]
    layout = {
        'xaxis': {
            'title': 'Instrumentalness',
            'zeroline': False,
        }
    }
    
    fig = go.Figure(data=graph,layout=layout)
    py.iplot(fig, filename = "Box Plot on Instrumentalness vs.Popularity")




if __name__ == "__main__":
    myData = loadData()
    acousticBoxplot(myData)
    instruBoxplot(myData)
       
    

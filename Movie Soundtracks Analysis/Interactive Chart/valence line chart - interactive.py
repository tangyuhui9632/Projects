#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 22:46:35 2018

@author: ly
"""

import pandas as pd 
import plotly.plotly as py
import plotly.graph_objs as go


def dataProcess():
    # Define desired columns
    sel_cols = ['Track_name','Movie_name','valence']
    # Read data frame and order the column as sel_cols
    df = pd.read_csv('cleaned_data.csv', sep=',',encoding='utf-8-sig',
                     usecols =sel_cols)[sel_cols]
    # Clean movie names, remove spaces
    df['Movie_name'] = [name.strip(' ') for name in list(df['Movie_name'])]
    
    return df


def subsetHarry(df):
    dflist = []
    # Find Harry Potter Series appeared in the data set
    filt = list(set([mv for mv in list(df['Movie_name']) if 'Harry Potter' in mv]))    
    
    for mv in filt:
        dflist.append(df[df['Movie_name'] == mv])
    
    return dflist
#%%
# For line and marker
def addTrace(df, color):
    
    trace = go.Scatter(
        x = df['Track_name'], 
        y = df['valence'],
        mode = 'lines+markers',
        visible = False,
        marker = dict(color = color),
        line = dict(color=color,width = 3),
        hovertext = df['Track_name'],
        hoverinfo = 'text',
        hoverlabel = dict(bgcolor = 'rgba(180, 180, 180,0.5)') 
        )

    return trace


# Add updatemenus
def addMenus():
    updatemenus = list([
        dict(type="buttons",
             active=-1,
             buttons=list([
                dict(label = '2004 The Prisoner of Azkaban',
                     method = 'update',
                     args = [{'visible': [True, False, False, False]}
                             ]),
                dict(label = '2005 The Goblet of Fire',
                     method = 'update',
                     args = [{'visible': [False, True, False,False]}
                              ]),
                dict(label = '2007 The Order of the Phoenix',
                     method = 'update',
                     args = [{'visible': [False, False, True, False]}
                             ]),
                dict(label = '2010 The Deathly Hallows',
                     method = 'update',
                     args = [{'visible': [False, False, False,True]},
                             ])
            ])
        )
    ])
    
    return updatemenus   
    

def plotLine(tracelist):
    data = tracelist
    # Edit the layout
    layout = dict(title = 'Music Valence and Movie Plots - Harry Potter Series',
                  xaxis = dict(title = 'Track Name',showticklabels=False),
                  yaxis = dict(title = 'Valence'),
                  updatemenus=addMenus()
                  )
    
    fig = dict(data=data, layout=layout)
    
    py.iplot(fig, filename='valence line interactive')

def main():
    # Filter out the original data set to obtain desired movie names
    df = dataProcess()
    # Get sub data frame lists
    dflist = subsetHarry(df)  

    colors = ['violet','salmon','palevioletred','red']
    tracelist = [addTrace(df,color) for df, color in zip(dflist, colors)]

    # Plot all     
    plotLine(tracelist)
    
if __name__ == "__main__":
    main()
    

    



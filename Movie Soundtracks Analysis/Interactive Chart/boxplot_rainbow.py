# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 22:57:30 2018

@author: youko
"""

import random
import plotly
import plotly.plotly as py
plotly.tools.set_credentials_file(username='youko', api_key='ITUYtqTHeahbeJXvhc6l')

from numpy import * 
import pandas as pd
import plotly.graph_objs as go

def yrgroup (row):
   return (row['Movie_yr']-1926)//3

def getplot1(data,N,c,traces0,namelist):
    
    # Each box is represented by a dict that contains the data, the type, and the colour. 
    # Use list comprehension to describe N boxes, each with a different colour and with different randomly generated data:
   
    for i in range(int(N)):
        gross=data[data['Yr_group']==i]
        gross=gross.drop_duplicates(subset='Movie_gross',keep='last')

        traces0.append(go.Box(
            y=gross['Movie_gross'].values,
            name=namelist[i],
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=c[i],
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
            ))

    layout = {'xaxis': {'showgrid':False,'zeroline':False, 'tickangle':60,'showticklabels':False,'title':'Movie Year'},
          'yaxis': {'zeroline':False,'gridcolor':'white','title':'Movie Gross'},
          'paper_bgcolor': 'rgb(233,233,233)',
          'plot_bgcolor': 'rgb(233,233,233)',
          'title':'boxplot for movie gross',
          }


    data=traces0
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='boxplot for movie gross')
    plt.clf()

def getplot2(data,N,c,traces0,namelist):
    
    # Each box is represented by a dict that contains the data, the type, and the colour. 
    # Use list comprehension to describe N boxes, each with a different colour and with different randomly generated data:
  
    for i in range(int(N)):
        popu=data[data['Yr_group']==i]
        #popu=gross.drop_duplicates(subset='popularity',keep='last')
        traces0.append(go.Box(
            y=popu['popularity'].values,
            name=namelist[i],
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=c[i],
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
         ))


    layout = {'xaxis': {'showgrid':False,'zeroline':False, 'tickangle':60,'showticklabels':False,'title':'Movie Year'},
          'yaxis': {'zeroline':False,'gridcolor':'white','title':'Soundtrack Popularity'},
          'paper_bgcolor': 'rgb(233,233,233)',
          'plot_bgcolor': 'rgb(233,233,233)',
          'title':'boxplot for soundtrack popularity',
          }


    data=traces0
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='boxplot for soundtrack popularity')
    plt.clf()
    
def getplot3(data,N,c,traces0,namelist):
    
    # Each box is represented by a dict that contains the data, the type, and the colour. 
    # Use list comprehension to describe N boxes, each with a different colour and with different randomly generated data:

    for i in range(int(N)):
        rating=data[data['Yr_group']==i]
        rating=rating.drop_duplicates(subset='Movie_name',keep='last')
        traces0.append(go.Box(
            y=rating['Movie_rate'].values,
            name=namelist[i],
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=c[i],
            marker=dict(
                size=2,
            ),
            line=dict(width=1),
          ))



    layout = {'xaxis': {'showgrid':False,'zeroline':False, 'tickangle':60,'showticklabels':False,'title':'Movie Year'},
          'yaxis': {'zeroline':False,'gridcolor':'white','title':'Movie Rating'},
          'paper_bgcolor': 'rgb(233,233,233)',
          'plot_bgcolor': 'rgb(233,233,233)',
          'title':'boxplot for movie rating',
          }


    data=traces0
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='boxplot for movie rating')
    plt.clf()

def main():
    data=pd.read_csv('cleaned_data.csv',sep = ',',encoding = 'utf-8-sig')
    data['Yr_group']  = data.apply (lambda row: yrgroup (row),axis=1)

    N = 31    # Number of boxes

    # generate an array of rainbow colors by fixing the saturation and lightness of the HSL representation of colour 
    # and marching around the hue. 
    # Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.
    c = ['hsl('+str(h)+',50%'+',50%)' for h in linspace(0, 360, N)]
    traces0 = []
    namelist=['1926','1929','1932','1935','1938','1941','1944','1947','1950','1953','1956','1959',
          '1962','1965','1968','1971','1974','1977','1980','1983','1986','1989','1992','1995',
          '1998','2001','2004','2007','2010','2013','2016']
    
    getplot1(data,N,c,traces0,namelist)
    getplot2(data,N,c,traces0,namelist)
    getplot3(data,N,c,traces0,namelist)
    
if __name__ == "__main__":
	 main()

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:44:07 2018

@author: youko
"""
##network for association rule
import plotly
plotly.tools.set_credentials_file(username='youko', api_key='ITUYtqTHeahbeJXvhc6l')
import plotly.plotly as py
import pandas as pd
import networkx as nx

# change the form of the results file

def arrangedata(data):
    data.columns = ["support", "ante", "conse"]
    cols = ['ante','conse','support']
    df = data[cols]
    # remove parentheses
    df['ante'] = df['ante'].apply(str).str.replace('\(|\)','')
    df['conse'] = df['conse'].apply(str).str.replace('\(|\)','')
    df= df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    #write the result to txt file
    return df

def makenet(df):
    G=nx.DiGraph()
    #G.add_nodes_from(df['ante'])
    #temp=zip(df['ante'],df['conse'],df['support'])
    #G.add_weighted_edges_from(temp)
    for i in range(13):
        G.add_edge(df['ante'][i],df['conse'][i],weight=df['support'][i])
        #give nodes names
    pos=nx.spring_layout(G)
    ##define the lists of  coordinates
    #set list of names
    labels=['Adventure','Action','Thriller','Crime','Mystery','Biography','Romance','Drama',
        'Comedy']
    Xn=[]
    Yn=[]
    for i in labels:
        Xn.append(pos[i][0])
        Yn.append(pos[i][1])
        # define the plotly trace for nodes
    trace_nodes=dict(type='scatter',
                 x=Xn, 
                 y=Yn,
                 mode='markers',
                 marker=dict(size=28, color='rgb(255,255,191)'),
                 text=labels,
                 hoverinfo='text')
    #record the coordinates of edge ends
    Xe=[]
    Ye=[]
    for e in G.edges():
        Xe.extend([pos[e[0]][0], pos[e[1]][0], None])
        Ye.extend([pos[e[0]][1], pos[e[1]][1], None])
    
    trace_edges=dict(type='scatter',
                 mode='lines',
                 x=Xe,
                 y=Ye,
                 line=dict(width=1, color='rgb(25,25,25)'),
                 text=list(df['support']),
                 hoverinfo='text' 
                )
    # plotly layout
    axis=dict(showline=False, # hide axis line, grid, ticklabels and  title
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title='' 
          )
    
    layout=dict(title= 'Association Rule Network',  
            font= dict(family='Balto'),
            width=600,
            height=600,
            autosize=False,
            showlegend=False,
            xaxis=axis,
            yaxis=axis,
            margin=dict(
            l=40,
            r=40,
            b=85,
            t=100,
            pad=0,
       
    ),
    hovermode='closest',
    plot_bgcolor='#efecea', #set background color            
    )
    fig = dict(data=[trace_edges, trace_nodes], layout=layout)
    py.iplot(fig)

def main():
    data = pd.read_csv('rule_net.txt', sep=",", header=None)
    df=arrangedata(data)
    makenet(df)


if __name__ == "__main__":
	main()

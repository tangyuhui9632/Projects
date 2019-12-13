#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 23:23:07 2018

@author: xuechunwang
"""

import plotly
import plotly.plotly as py
import pandas as pd
import plotly.graph_objs as go
from os import path
import matplotlib.pyplot as plt
import numpy as np

def main():
    #Set up credentials
    plotly.tools.set_credentials_file(username='xw240', api_key='khaHi0IaklUF3Of2RUNU')
    #read the data 
    myData = pd.read_csv('cleaned_data.csv', sep=',',encoding='utf-8-sig')
    #change the data type of attributes for plotting
    # myData.Movie_gross = myData.Movie_gross.str.extract('(\d+\.\d+)').astype(np.float64)
    # myData.Movie_runtime = myData.Movie_runtime.str.extract('(\d+)').astype(np.float)
    # myData.Movie_yr = myData.Movie_yr.str.extract('(\d+)').astype(np.float)
    myData.Movie_gross.replace(np.nan,0,inplace=True)
    #run the following functions to make plot
    hist_inst(myData)
    hist_year(myData)
    hist_low_instu_count(myData)
    scatter_popularity(myData)
    scatter_feature(myData)
    word_cloud(myData)
    
    
#This function will plot the Histogram of Popularity with different instrumentalness groups
def hist_inst(myData):
    #seperate the data set as high instrumentalness tracks and low instrumentalness tracks
    trackData_inst_1 = myData[(myData['instrumentalness'] <= 0.5)]
    trackData_inst_2 = myData[(myData['instrumentalness'] > 0.5)]
    #the the trace data set 1
    trace1 = go.Histogram(
        x=trackData_inst_1.popularity,
        histnorm='percent',
        name='Low instrumental',
        opacity=0.75,
        xbins=dict(
            start=-1,
            end= 80,
            size=10
        )
    )
    #the the trace data set 2
    trace2 = go.Histogram(
        x=trackData_inst_2.popularity,
        histnorm='percent',
        name='High instrumental',
        opacity=0.75,
        xbins=dict(
            start=-1,
            end= 80,
            size=10
        )
    )
    #set the data for the plot
    data = [trace1, trace2]   
    layout = go.Layout(
        title='Histogram of Popularity',
        xaxis=dict(
            title='Popularity Value'
        ),
        yaxis=dict(
            title='percentage'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    #set up the figure
    fig = go.Figure(data=data, layout=layout)
    #create the plot to my account 
    py.iplot(fig, filename='Overlaid Histogram of Popularity 2')

#This function will plot the hist plot of how many movie use a certain number of low instrumrnal songs
def hist_low_instu_count(myData): 
    #goupe the data with album id
    groupedDF = myData.groupby("Album_ID")
    low_instru_count = []
    total_count = []
#    popularity_max = []
#    gross = []
    #collect the max mean popularity of tracks in each album along with the movie gross
    for albumID,group in groupedDF:
        count = 0
        for i in range(0, len(group)):
            if(group.iloc[i]['instrumentalness'] <= 0.05):
                count = count + 1
        low_instru_count.append(count)
        total_count.append(len(group))
        #popularity_max.append(group.max()["popularity"])
        #gross.append(group.mean()["Movie_gross"])
        
       #the the trace data set 1
    trace1 = go.Histogram(
        x=low_instru_count,
        histnorm='percent',
        name='Low instrumental',
        opacity=0.75,
        xbins=dict(
            start=-1,
            end= 30,
            size=1
        )
    )
    #set the data for the plot
    data = [trace1]   
    layout = go.Layout(
        title='Histogram of Low Instrumentalness Count',
        xaxis=dict(
            title='Popularity Value'
        ),
        yaxis=dict(
            title='percentage'
        ),
    )
    #set up the figure
    fig = go.Figure(data=data, layout=layout)
    #create the plot to my account 
    py.iplot(fig, filename='Histogram of Low Instrumentalness Count')
    
    trace1 = go.Histogram(
        x=total_count,
        histnorm='percent',
        name='Total Track Count',
        opacity=0.75,
        xbins=dict(
            start=-1,
            end= 30,
            size=1
        )
    )
    #set the data for the plot
    data = [trace1]   
    layout = go.Layout(
        title='Histogram of Total Track Count',
        xaxis=dict(
            title='Total Track Count'
        ),
        yaxis=dict(
            title='percentage'
        ),
    )
    #set up the figure
    fig = go.Figure(data=data, layout=layout)
    #create the plot to my account 
    py.iplot(fig, filename='Histogram of Total Track Count')
    


#This function will plot the Histogram of Popularity with different year groups
def hist_year(myData):
    #break the data into 3 groups with different range of years
    trackData_09_18 = myData[(myData['Movie_yr'] <= 2018) & (myData['Movie_yr'] >= 2009 )]
    trackData_99_08 = myData[(myData['Movie_yr'] <= 2008) & (myData['Movie_yr'] >= 1999 )]
    trackData_89_98 = myData[(myData['Movie_yr'] <= 1998) & (myData['Movie_yr'] >= 1989 )]
    #the the trace data set 1
    trace1 = go.Histogram(
        x=trackData_09_18.popularity,
        name='year 2009-2018',
        opacity=0.75,
        xbins=dict(
            start=-1,
            end= 80,
            size=10
        )
    )
    #the the trace data set 2
    trace2 = go.Histogram(
        x=trackData_99_08.popularity,
        name='year 1999-2008',
        opacity=0.75,
        xbins=dict(
            start=-1,
            end= 80,
            size=10
        )
    )
    #the the trace data set 3
    trace3 = go.Histogram(
        x=trackData_89_98.popularity,
        name='year 1989-1998',
        opacity=0.75,
        xbins=dict(
            start=-1,
            end= 80,
            size=10
        )
    )
    data = [trace1, trace2, trace3]
    #set up the layout of the plot
    layout = go.Layout(
        barmode='overlay',
        title='Histogram of Popularity',
        xaxis=dict(
            title='Popularity Value'
        ),
        yaxis=dict(
            title='Count'
        ),
        bargap=0.2,
        bargroupgap=0.1
    )
    #set up the figure
    fig = go.Figure(data=data, layout=layout)
    #create the plot to my account 
    py.iplot(fig, filename='Overlaid Histogram of Popularity 1')

#This function will plot the Scatted plot of max album Popularity and movie gross
def scatter_popularity(myData): 
    #goupe the data with album id
    groupedDF = myData.groupby("Album_ID")
    popularity_mean = []
    popularity_max = []
    gross = []
    rate = []
    albumIDs = []
    #collect the max mean popularity of tracks in each album along with the movie gross
    for albumID,group in groupedDF:
        if(group.mean()["Movie_gross"] > 0):
            albumIDs.append(albumID)
            popularity_mean.append(group.mean()["popularity"])
            popularity_max.append(group.max()["popularity"])
            gross.append(group.mean()["Movie_gross"])
            rate.append(group.mean()["Movie_rate"])   
    #set up the trace of the data
    trace = go.Scatter(
        x = gross,
        y = popularity_max,
        mode = 'markers'
    )
    data = [trace]
    #set up the layout 
    layout = go.Layout(
        barmode='overlay',
        title='Scatter Plot of Popularity and Movie Gross',
        xaxis=dict(
            title='Gross'
        ),
        yaxis=dict(
            title='Popularity'
        ),
    )   
    # Plot and embed in my account
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='Scatter Plot of Popularity and Movie Gross')

#This function will plot the 3D Scatted plot of track features and create the bubble chart of the track feature
#This will only plot for the track with popularity over 70
def scatter_feature(myData):  
    track_popular = myData[(myData['popularity'] >= 70)]   
    #set up the trace of the data
    trace1 = go.Scatter3d(
        x=track_popular['energy'],
        y=track_popular['acousticness'],
        z=track_popular['valence'],
        mode='markers',
        marker=dict(
            size=12,
            # set color to an array/list of desired values
            color=track_popular['time_signature'],  
            # choose a colorscale              
            colorscale='Viridis',
            opacity=0.8
        )
    )
    data = [trace1]
    #set up the layout
    layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene = dict(
    xaxis = dict(
        title='energy'),
    yaxis = dict(
        title='acousticness'),
    zaxis = dict(
        title='valence'),),


    )
    #set up the figure
    fig = go.Figure(data=data, layout=layout)
    #plot to plotly account
    py.iplot(fig, filename='3D Scatter Plot of Track Features')
    
    #create the bubble chart of the track feature
    data = [
        {
            'x': track_popular['danceability'],
            'y': track_popular['energy'],
            'mode': 'markers',
            'marker': {
                'color':track_popular['time_signature'] ,
                'size': track_popular['key'],
                'showscale': True
            }
        }
    ]
    #set up the layout
    layout = go.Layout(
        title='Danceability Vs Energy',
        xaxis=dict(title='Danceability'),
        yaxis=dict(title='Energy')
    )
    #set up the figure
    fig2 = go.Figure(data=data, layout=layout)
    py.iplot(fig2, filename='Scatter Plot of Track Features')  
    
    
#This function will plot the word cloud for track names and movie names
def word_cloud(myData):
    #import the package that will be uesd
    from wordcloud import WordCloud, STOPWORDS
    from PIL import Image
    import nltk
    #opent the mask picture for backgroud mask
    cloudmask = np.array(Image.open(path.join("mask-cloud.jpg")))
    text = " ".join(list(myData['Track_name']))
    text = text.lower()
    stopwords = set(STOPWORDS)
    #append stop words
    stopword = nltk.corpus.stopwords.words('english')
    stopword.append("drink")
    stopword.append("now")
    stopword.append("feat")
    stopword.append("got")
    stopword.append("song")
    stopword.append("soundtrack")
    stopword.append("version")
    stopword.append("track")
    stopword.append("instrumental")
    stopword.append("remastered")
    stopword.append("theme")
    newtext1 = text.split(' ')
    newlist = [w for w in newtext1 
                           if w not in stopword 
                           and w.isalpha()]
    fdist1 = nltk.FreqDist(newlist)
    print (fdist1.most_common(50))
    fdist1.plot(30,cumulative=False)
    stopwords.update(["drink", "now", "feat", "got", "song", "soundtrack", "version","track","instrumental","remastered","theme","remix"])
    #extract the word cloud from track name text
    wordcloud = WordCloud(width=3765, height=2345,background_color="white",
                          stopwords=stopwords,mask = cloudmask).generate(text)
    #set up the figure
    plt.figure( figsize=(25,15))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    #save it to a file
    plt.savefig("track word cloud.jpg")
    plt.show()
    groupedDF = myData.groupby("Movie_name")
    movie_names = []
    for Movie_name,group in groupedDF:
        movie_names.append(Movie_name)
    text = " ".join(list(movie_names))
    text = text.lower()
    newtext2 = text.split(' ')
    newlist = [w for w in newtext2 
                           if w not in stopword 
                           and w.isalpha()]
    fdist1 = nltk.FreqDist(newlist)
    print (fdist1.most_common(50))
    fdist1.plot(30,cumulative=False)
    #extract the word cloud from movie name text
    wordcloud = WordCloud(width=3765, height=2345,background_color="white",
                          stopwords=stopwords,mask = cloudmask).generate(text)
    #set up the figure
    plt.figure( figsize=(25,15))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig("movie word cloud.png")
    plt.show()
 

if __name__ == "__main__":
	main()
    
    

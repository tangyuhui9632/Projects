# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 21:40:54 2018

@author: Yuhui Tang
"""


# Import Libraries
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind
from sklearn.preprocessing import normalize
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS
import statsmodels.api as sm 


# Read data and select only non_numerical variables to prepare for t-test
def data_processing():
    # Read data frame
    myData = pd.read_csv('cleaned_data.csv', sep = ',',encoding = 'utf-8-sig')
    # Select only numeric columns
    numericDF = myData.select_dtypes(include = np.number)
    # Normalize the data frame and return it 
    return pd.DataFrame(normalize(numericDF), columns = numericDF.columns)


# Conduct t test
def ttest(x, y):
    ttest = ttest_ind(x, y)
    print(ttest)

# Conduct linear regression
def linear_reg(X,y):
    X = sm.add_constant(X) ## Add an intercept (beta_0) to our model
    model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    print(model.summary()) # Print out the statistics
    
def main():
    # Process the data
    normalizedData = data_processing()
#    mode_value = normalizedData['mode'].values
#    energy_value = normalizedData['energy'].values
#    danceability_value = normalizedData['danceability'].values
#    
#    # T-test on two groups: mode&energy, mode&danceability
#    ttest(mode_value,energy_value)
#    ttest(mode_value,danceability_value)
    
    # Linear Regression on how popularity is related to valence, movie_gross, movie rate
    X, y = normalizedData[['valence','Movie_gross','Movie_rate']], normalizedData.popularity
    linear_reg(X,y)
    
if __name__ == "__main__":
    main()
    
    
    
    

    
    
    
    

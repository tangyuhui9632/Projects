#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:00:52 2018
@author: xuechunwang
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from scipy import interp
import matplotlib.pyplot as plt

#Read data and delete non_numerical varivales to prepare for prediction


def data_processing():
    myData = pd.read_csv('cleaned_data.csv', sep=',',encoding='utf-8-sig')
    # Select only numeric columns
    numericDF = myData.select_dtypes(include = np.number)
    # Make a histogram for the popularity column
    # numericDF.popularity.hist()
    # From the histogram, we can observe that 40 is a good point to break down and make the bins
    binnames = [0,1]
    bins1=[-0.1,40,100] # extend the lower bound from 0 to -0.1 to include the minimum value of popularity
    popGroups = pd.cut(numericDF['popularity'], bins1, labels=binnames)

#    popGroups =  pd.to_numeric(popGroups, downcast='integer')

    norm_myData = pd.DataFrame(normalize(numericDF),columns = numericDF.columns)
    # add popGroups column to the normalized data frame
    norm_myData['PopGroups'] = popGroups
    return norm_myData



# Use statistical test to help us elminate useless features to simplify our model
def feature_selection(X,Y,number):
    # feature extraction using chi test
    test = SelectKBest(score_func=chi2, k=number)
    fit = test.fit(X, Y)
    # summarize scores of each variable except for the predicting variable
    np.set_printoptions(precision=3)
    print(fit.scores_)
    features = fit.transform(X)
    # return the selected features for future training
    return features


# Use cross validation to plot the ROC curve for each training model independently and draw all
# curves in one plot for better comparison
def cross_validation(models,X,Y):
    # We will conduct a 5-fold cross validation
    cv = StratifiedKFold(n_splits=5)
    mean_fpr_list = []
    mean_tpr_list = []
    for name, model in models:
        true_posit_set = []
        accuracy_set = []
        mean_fpr = np.linspace(0, 1, 100)
        for train, test in cv.split(X, Y):
            # fit the models 
            probas = model.fit(X[train], Y[train].ravel()).predict_proba(X[test])
            # Compute ROC curve and collected the false positive, true positive,... rate 
            fpr, tpr, thresholds = roc_curve(Y[test], probas[:, 1])
            true_posit_set.append(interp(mean_fpr, fpr, tpr))
            true_posit_set[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            accuracy_set.append(roc_auc)
            
        plt.figure()
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        #calculate the mean of test results of all validations
        mean_tpr = np.mean(true_posit_set, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        # Also calculate the standard deviation
        std_auc = np.std(accuracy_set)
        mean_fpr_list.append(mean_fpr)
        mean_tpr_list.append(mean_tpr)
        #plot the mean of a collection of false positive rate, true positive rate   
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)        
        std_tpr = np.std(true_posit_set, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        titlename = 'Receiver operating characteristic for ' + name
        plt.title(titlename)
        plt.legend(loc="lower right")
        plt.show()
    return(mean_fpr_list,mean_tpr_list)


#This function is just trying to plot all ROC curve generated in the last function into the same plot
def plot_in_one(mean_fpr_list,mean_tpr_list,models):
    plt.figure(figsize=(10,10))
    i = 0
    lw = 2
    colorlist = ['darkorange','red','blue','green','blueviolet','pink','gold']
    for name, model in models:
        plt.plot(mean_fpr_list[i], mean_tpr_list[i], color= colorlist[i],lw=lw,label= name)
        i += 1
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Plot comparison')
    plt.legend(loc="lower right")
    plt.show()  
    

#Use the cleaned data set to split into train and test data to train and test the models.
def classifier(models,X,Y):
    test_size = 0.20
    # Set the threshold value as 0.5, but you can change it to any confidence leverl you want.
    threshold = 0.5
    #seed = 7
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size)
    predictlist = []
    names = []
    # train the data with all models in the models set 
    for name, model in models:
        # Use the model to predict the test features 
        probability_result = model.fit(X_train,Y_train.ravel()).predict_proba(X_validate)
        predict = (probability_result [:,1] >= threshold).astype('int')
        predictlist.append((predict,name))
        names.append(name)
    #calculate the confusion matrix and accuracy score
    for predict, name in predictlist:
        print("\nTest Accuracy for " + name + ":", accuracy_score(Y_validate, predict))
        report = classification_report(Y_validate, predict)
        print(report)
        print("The confusion matrix:")
        matrix = confusion_matrix(Y_validate, predict)
        print(matrix)

def addModels():
    #initialize a set of models to train 
    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 20)))
    models.append(('DecisionTree', DecisionTreeClassifier()))
    models.append(('NaiveBayes', GaussianNB()))
    models.append(('SVM', svm.SVC(kernel='rbf',probability = True)))
    models.append(('RandomForest', RandomForestClassifier(n_estimators=100, random_state=123)))
    return models

def main():

    myData = data_processing() 
    cols = [col for col in myData.columns if col not in ['PopGroups', 'popularity']]
    # Seperate the train feature data and label data 
    Xdata = myData[cols]
    Ydata = myData.loc[:, myData.columns == 'PopGroups']
    Xdata['loudness'] = Xdata['loudness'].abs()
    Y = Ydata.values
    X = Xdata.values
    c, r = Y.shape
    Y = Y.reshape(c,)
    print("\nSelect KBest test scores:")
    print(Xdata.columns)
    newX = feature_selection(X,Y,7)
    models = addModels()
    mean_fpr_list,mean_tpr_list = cross_validation(models,newX,Y)
    plot_in_one(mean_fpr_list,mean_tpr_list,models)
    classifier(models,newX,Y)
    
if __name__ == "__main__":
	main()

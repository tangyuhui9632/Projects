# -*- coding: utf-8 -*-
"""
Created on May 10 

@author: Charlotte Tang

version of submission (v20.01)

candidate ID (C1552034)
"""


################################### Pre-processing Functions ##############################################################
import pandas as pd
import numpy as np

import itertools
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency

from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score 

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.metrics import roc_auc_score, make_scorer,f1_score,recall_score

from sklearn import metrics

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score, make_scorer
from pprint import pprint


######################### Functions for Analysis ##################################
def percent_missing_values(df):
    percent_missing = df.isnull().sum() * 100 / df.shape[0]
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    missing_value_df = missing_value_df.sort_values('percent_missing',ascending = False)

    return missing_value_df

def datatype_summary(df):
    #different datatype:
    all_null_feature=[] #columns full with missing values
    most_null_feature=[] #columns with 0% of missing values
    num_feature=[] #numerical columns
    object_feature=[] #categorical columns
    bool_feature=[] #boolean features

    #Let's categories features into numerical, objective types.
    for col in df.columns:
        if df[col].isnull().sum()==df.shape[0]: #features having 100% of missing values 
            all_null_feature.append(col)
        else:
            if df[col].isnull().sum()>(len(df)*0.8): #features having more than 70% of missing values
                most_null_feature.append(col)
            if df[col].dtype in ('int' , 'float64', 'float32'): #features is objective
                num_feature.append(col)
            elif df[col].dtype == 'object': #features is numerical
                object_feature.append(col)
            elif df[col].dtype == 'bool':
                bool_feature.append(col)                      
    
    print("The columns with all NULL values are: " , all_null_feature)          
    print("The columns with 80% NULL values are: " , most_null_feature)                                   
    print("The numerical columns are: " , num_feature)                                   
    print("The object columns are: " , object_feature)     
    print("The boolean columns are: " , bool_feature)       

######################### Plotting Functions ######################################
# Define functions
def bar_plot(df,column, x, y,title, xlabel, ylabel):
    
    labels = list(df[column])

    colors = {'online_gifts':'skyblue','online_retail': 'thistle', 'rideshare': 'pink',
              'airline':'orange', 'fastfood': 'yellowgreen', 'entertainment':'navajowhite', 'food':'grey'}
    ax = df.plot(kind='barh', x=x,y=y,figsize=(8, 5),color=[colors[i] for i in labels],
                  width = 0.7)
    ax.set_title(title,fontsize=18)
    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    patchList = []
    for key in colors:
            data_key = mpatches.Patch(color=colors[key], label=key)
            patchList.append(data_key)

    ax.legend(handles=patchList,prop={'size': 15})
    plt.show()
    
def pie_chart(df, df1,target_col, colors1, colors2):
    label=df.groupby(target_col).size().index
    sizes=df.groupby(target_col).size().values

    label1=df1.groupby(target_col).size().index
    sizes1=df1.groupby(target_col).size().values

    colors1 = colors1
    colors2 = colors2

    # make figure and assign axis objects
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fig.subplots_adjust(wspace=0)

    # large pie chart parameters
    # rotate so that first wedge is split by the x-axis
    ax1.pie(sizes, autopct='%1.1f%%', labels=label,colors=colors1,textprops={'fontsize': 10})
    ax2.pie(sizes1, autopct='%1.1f%%',labels=label1, colors=colors2,textprops={'fontsize': 10})

    # set titles
    ax1.set_title('Total Transactions by ' + target_col,fontsize=16)
    ax2.set_title('Multi-swipe Transactions by ' + target_col,fontsize=14)

    
    plt.show()
    
def histogram(col,df1,df2, label1,label2, size ):
    
    plt.subplots(figsize=size, dpi=100)
    sns.distplot(df1[col] , color="skyblue", label=label1,hist_kws=dict(alpha=0.5),bins=30)
    if df2 is not None:
        sns.distplot(df2[col] , color="pink", label=label2, hist_kws=dict(alpha=0.5),bins=30)
    plt.title('Frequency Histogram of %s by isFraud'%(col))
    plt.xlabel(col)
    plt.ylabel('Percentage')
    plt.legend();
    # Add labels
    plt.title('Histogram of  %s by isFraud'%(col),fontsize = 22)
    plt.xlabel(col,fontsize=15)
    plt.ylabel('Percentage',fontsize=15)
    plt.show()

def distribution_plot(col, data, title, Y, overlapping, fig_size):
        fig,(ax1) = plt.subplots(figsize=fig_size)
        total = len(data)
        orders = list(set(data[col]))
        palette = ['pink', 'skyblue']
        ax1 = sns.countplot(x = col, data = data, palette = palette, hue = Y, order = orders)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax1.set_xlabel( col, fontsize=10)
        ax1.set_ylabel('number of %s'%(col), fontsize=12)
        ax1.set_title('Countplot of %s in %s'%(col,title) , fontsize=15)
        
        sizes=[]
        for p in ax1.patches:
            height = p.get_height()
            sizes.append(height)
            ax1.text(p.get_x() + p.get_width()/2.,\
                     height + 3,'{:1.2f}%'.format(height/total*100),\
                     ha='center', fontsize=10) 

        plt.show()
        

############################### Data Manipulation Functions #################################
def target_variable_summary(df, col, population, labels):
    classes=df[col].value_counts()
    normal=round(classes[0]/df[col].count()*100,2)
    fraud=round(classes[1]/df[col].count()*100,2)
    
    print("For the {}, Non-Fraud is {}%".format(population, normal))
    print("For the {}, Fraud is {}%".format(population, fraud))
    
    plt.figure(figsize=(20,6))
    plt.subplot(1,2,1)
    ax=sns.countplot(df[col])
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
    plt.ylabel("Number of transaction")
    plt.xlabel(col)
    plt.title("Credit Card Fraud Class - data unbalance")
    plt.grid()
    plt.subplot(1,2,2)
    fraud_percentage = {col:labels, 'Percentage':[normal, fraud]} 
    df_fraud_percentage = pd.DataFrame(fraud_percentage) 
    ax=sns.barplot(x=col,y='Percentage', data=df_fraud_percentage)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
    plt.title('Percentage of fraud vs non-fraud transcations for {}'.format(population))

    plt.grid()
    
def remove_missing_values(df, threshold):
    '''
    This function removes columns with percentage of missing value over a certain threshold
    '''
    # check percentage of missing value value in each column
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
    missing_value_df = missing_value_df.sort_values('percent_missing',ascending = False)

    # remove columns has over the threshold of missing value
    col_remove_df = missing_value_df[missing_value_df.values > threshold].index
    df = df.loc[:,~df.columns.isin(col_remove_df)]
    return df

def check_unique_value(df):
    '''
    This function check unique values of each column.
    It removes the columns with all same values.
    '''
    # check number of unique values in each column
    df_unique = df.apply(lambda x: x.nunique()).sort_values(axis=0, ascending=False)
    df_unique = pd.DataFrame({'unique_value_num': df_unique})
    df_unique = df_unique.sort_values('unique_value_num',ascending = False)

    # remove columns has only 1 unique value (has all same number)
    col_remove_df = df_unique[df_unique.values == 1].index
    df = df.loc[:,~df.columns.isin(col_remove_df)]
    return df

def distribution(df, col):
    '''
    This function checks the distribution of a column
    using histogram and boxplot
    '''
    fig, ax = plt.subplots(1,2, figsize=(15,4))
    sns.distplot(df[col], ax=ax[0], color= 'orange')
    sns.boxplot(df[col], ax=ax[1], color = 'pink')
    fig.suptitle('Distribution of '+ col)
    fig.show()

def encode_and_bind(df, feature_to_encode):
    '''
    The function encodes categorical variables
    '''
    dummies = pd.get_dummies(df[[feature_to_encode]])
    
    res = pd.concat([df, dummies], axis=1)
    res.drop([feature_to_encode],axis = 1, inplace = True)
    return res

def high_correlation(df, col_list, threshold):
    '''
    This function calculates correlation among certain columns.
    It removes those columns with high correlation with another
    '''
    corr_matrix = df.loc[:,col_list].corr()
    high_corr_var = np.where(corr_matrix > threshold)
    high_corr_var=[(corr_matrix.index[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x != y and x < y]
    col_remove = [unit[0] for unit in high_corr_var]
    df.drop(col_remove, inplace=True, axis = 1)
    return df

def impute_missing_value(df, strategy):
    # Impute the missing values with median
    impute = SimpleImputer(missing_values=np.nan, strategy = strategy)
    temp = df
    impute.fit(temp)
    temp=impute.transform(temp)
    temp = pd.DataFrame(temp, columns= df.columns)
    df = temp
    
    return df

def standardization(train, test):
    scale = StandardScaler()
    oversampled_features_df = scale.fit_transform(train)
    test_features = scale.transform(test)

    return train, test

def chi2(df, col1, col2):
    CrosstabResult=pd.crosstab(index=df[col1],columns=df[col2])
    # Performing Chi-sq test
    ChiSqResult = chi2_contingency(CrosstabResult)

    # P-Value is the Probability of H0 being True
    # If P-Value>0.05 then only we Accept the assumption(H0)

    print('The P-Value of the ChiSq Test for %s and %s is:'%(col1,col2), ChiSqResult[1])

################################### Modeling Functions ##############################################################

def train_test_splits(df, target, group1, group2, fraction):
    # seperate train and test by 0.7: 0.3
    target_yes = df[df[target] == group1]
    target_no = df[df[target] == group2]
    target_yes_train = target_yes.sample(frac=fraction)
    target_no_train = target_no.sample(frac=fraction)
    train = pd.concat([target_yes_train, target_no_train], axis=0)
    train_features = train.loc[:, df.columns != target]
    train_target = train[target]
    
    test = df[~df.isin(train)].dropna()
    test_features = test.loc[:, test.columns != target]
    test_target = test[target]
    
    return train, test, train_features, train_target, test_features, test_target


def oversample(df, train_x, train_y, variable, sample_size, random_state, k_neighbors):
    # Oversample using SMOTE on the train set
    
    smote = SMOTE(sampling_strategy=sample_size, random_state=random_state,  k_neighbors=k_neighbors)
    oversampled_features, oversampled_target = smote.fit_sample(train_x, train_y)
    
    oversampled_features_df = pd.DataFrame(oversampled_features, columns= train_x.columns)
    oversampled_target_df = pd.DataFrame(oversampled_target, columns= [variable])
    train_oversampled = pd.concat([oversampled_features_df, oversampled_target_df], axis=1)
    
    return train_oversampled, oversampled_features_df, oversampled_target_df

def DecisionTree(X_train, Y_train, X_test,Y_test):
    parameters = {'max_depth': (3,5),
                  'max_features': (5,7),
                  'min_samples_leaf': (5,10),
                  'criterion': ("gini", "entropy")}


    model = DecisionTreeClassifier()
    grid_obj = GridSearchCV(model, param_grid=parameters, scoring = make_scorer(f1_score), 
                            verbose=1, n_jobs=4, cv=4)
    grid_obj = grid_obj.fit(X_train, Y_train)
    estimator = grid_obj.best_estimator_
    print(estimator)
    return estimator

def RandomForest(X_train, Y_train, X_test,Y_test):
    class_weight = dict({1:90, 0:10})

    parameters = {
#                   'n_estimators':(50,100),
                  'criterion':('gini','entropy'),
                  'min_samples_split':(2,4),
                  'min_samples_leaf':(1,3)}

    model = RandomForestClassifier(n_estimators = 200, class_weight=class_weight)
    grid_obj = GridSearchCV(model, param_grid=parameters, cv=5,scoring='roc_auc', verbose=1, n_jobs=4) 
    grid_obj = grid_obj.fit(X_train, Y_train)
    estimator = grid_obj.best_estimator_
    print(estimator)
    return estimator




def model_pred(alg, train_x, train_y, test_x, test_y,threshold, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    #Predict training and test set:
    dtrain_predictions = alg.predict(train_x)
    dtest_predictions = alg.predict(test_x)
    
    dtrain_predprob = alg.predict_proba(train_x)[:,1]
    dtest_predprob = alg.predict_proba(test_x)[:,1]
    
    y_pred_train = (dtrain_predprob >= threshold).astype(bool)
    y_pred_test = (dtest_predprob >= threshold).astype(bool)
    
    # compare with threshold 
    y_pred_over_threshold = (dtest_predprob >= threshold).astype(bool)
    
    return dtrain_predprob, dtest_predprob, dtest_predictions, y_pred_over_threshold


# Create a confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def confusion_matrix(train_y, test_y, dtrain_predprob, dtest_predprob, dtest_predictions, y_pred_over_threshold):
    # metrics
    
    precision = round(metrics.precision_score(test_y, dtest_predictions),4)
    recall = round(metrics.recall_score(test_y, dtest_predictions),4)
    f1_score = round(metrics.f1_score(test_y, dtest_predictions),4)
    
    #confusion_matrix
    cnf_matrix = metrics.confusion_matrix(test_y, y_pred_over_threshold)
    tn, fp, fn, tp = metrics.confusion_matrix(test_y, y_pred_over_threshold).ravel()
    
    # more metrics
    specificity = round(tn / (tn+fp), 4)
    fpr = round(fp / (tn + fp),4)
    fnr = 1 - recall
    
    # plot confusion matrix
    labels = ['No Fraud', 'Fraud']
    fig = plt.figure(figsize=(15,8))   
    plot_confusion_matrix(cnf_matrix, labels, title="Confusion Matrix for test set", cmap=plt.cm.Oranges)

    
    print('\n')
    
    print("Evaluation Metrics for the Model on Test set is: \n")
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1_score)
    print("False Positive Rate:", fpr)
    print("False Negative Rate:", fnr)
    print('\n')
    
    
def ROC(train_y, test_y, dtrain_predprob, dtest_predprob, dtest_predictions, y_pred_over_threshold):

    print('The ROC curves for train and test set are:')
    # ROC curve
    fpr_0, tpr_0, thresholds_0 = roc_curve(train_y, dtrain_predprob)
    fpr_1, tpr_1, thresholds_1 = roc_curve(test_y, dtest_predprob)
    roc_auc_0 = auc(fpr_0, tpr_0)
    roc_auc_1 = auc(fpr_1, tpr_1)
    print("AUC for train set: %f" % roc_auc_0)
    print("AUC for test set: %f" % roc_auc_1)
    
    # Plot ROC curve
    plt.figure(figsize=(8,8))
    plt.plot(fpr_0, tpr_0, label='ROC curve - train(AUC = %0.2f)' % roc_auc_0, color='blue')
    plt.plot(fpr_1, tpr_1, label='ROC curve - test (AUC = %0.2f)' % roc_auc_1, color='orange')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves ')
    plt.legend(loc="lower right")
    plt.show()
        

def feature_importance(model_name, model, df, top_n):
    if model_name == 'rf':
        coefficients = model.feature_importances_.tolist()
    elif model_name == 'lg':
        coefficients = abs(model.coef_)[-1]
    temp = list(zip(coefficients, list(df.columns)))
    coef = [item[0] for item in temp]
    feature = [item[1] for item in temp]
    feature_importance = pd.DataFrame({'coef': coef,
                                   'feature': feature}).sort_values('coef', ascending = False)[:top_n]
    f = sns.factorplot('coef','feature', data=feature_importance, kind="bar",palette="Blues",size=6,aspect=2,legend_out=True)
    f.set_xlabels('Feature Importnce', fontsize= 15)
    f.set_ylabels('Features', fontsize= 15)
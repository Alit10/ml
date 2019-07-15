# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:13:07 2019

@author: ali.tber
"""

import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt

%matplotlib inline
%load_ext autoreload
%autoreload 2


def grid_search(clf,x,y, cv =2 , n_jobs =2):
    rf_params = { 
            'n_estimators': [200, 700],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth':np.arange(5,20,10)
            }
    gs_clf = GridSearchCV(estimator= clf, param_grid=rf_params,scoring = make_scorer(accuracy_score),  
                          cv= cv, n_jobs = n_jobs )
    print(type(x))
    print(type(y))
    print(type(gs_clf))
    gs_clf.fit(x,y)
    print(gs_clf.best_params_)
    return gs_clf

df = pd.read_csv("df_train_ali.csv",sep=";",nrows = 10000)


params = {"criterion" : "gini", "max_depth" : None,"min_samples_split" : 2, 
          "min_samples_leaf" : 1, "min_weight_fraction_leaf" :0.0, 
          "max_features" : "auto", "max_leaf_nodes" : None, 
          "min_impurity_decrease":0.0, "oob_score" : True , "verbose" : 1}


n_estimators = 100
clf_rf = RandomForestClassifier(n_estimators = n_estimators, **params)

clf_rf.fit(df.drop(["Spec_group"], axis = 1) , df.Spec_group)
clf_rf.oob_score_


rs = grid_search(clf_rf,x=df.drop(["Spec_group"], axis = 1) , y=df.Spec_group )






class RandomF():
    def __init__(self, clf,x_train,x_test,y_train,y_test):
        self.clf = clf
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
    
    def fit():
        
        
        















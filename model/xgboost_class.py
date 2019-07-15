# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:04:39 2019

@author: ali.tber
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt

%matplotlib inline
%load_ext autoreload
%autoreload 2


df = pd.read_csv("../data/Xy_train_2.csv")





#apply xgboost 

params = {"n_estimators" =n_estimators,
                        "max_depth" = 4,
                        "objective"= "multi:softprob",
                        "learning_rate" = .05, 
                        "subsample" = 0.8, 
                        "colsample_bytree"= .8,
                        "gamma" = 1,
                        "reg_alpha" = 0,
                        "reg_lambda" = 1,
                        "nthread" = 4,
                       "min_child_weight" = 10,
                       "silent" = 1}


clf_xgb = XGBClassifier( **params, eval_metric = "merror")


#Methode pour sortir une prédiction avec un kflod
folds = StratifiedKFold(n_splits=10, shuffle=False, random_state=44000)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    print("Fold {}".format(fold_))
    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    num_round = 1000000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3000)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))





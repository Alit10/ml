# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:34:19 2019

@author: ali.tber
"""
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,roc_auc_score,roc_curve
from scikitplot.metrics import plot_cumulative_gain


def plot_roc(ytrue,ypred):
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(ytrue, ypred)
    roc_auc = roc_auc_score(ytrue, ypred)
    print(roc_auc)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

def plot_lift(y_true,y_pred):
    plot_cumulative_gain(y_true, y_pred)
    plt.show()

#Multilabel classification
if __name__=="main":
    print(classification_report(y_true,y_pred))
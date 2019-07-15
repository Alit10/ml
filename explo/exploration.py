"""
@author = Ali 
date = 21-05-2019
"""
import pandas as pd 
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

def summary(df,max_number_occ = 100):
    """
    Takes a dataframe and return the 5 most occcurences on the object type columns
    """
    l = []
    df2 = pd.DataFrame({"col1":np.arange(5)})
    for col in df.columns:
        if df[col].dtypes == "object":
            if df[col].nunique() > max_number_occ :
                l.append(col)
            else : 
                df_temp = pd.DataFrame((df[col].value_counts(dropna=False)/df.shape[0]).reset_index()).rename(columns = {
        col:"pct_"+col,"index":col})
                df_temp = df_temp.sort_values(by="pct_"+col,ascending=False).iloc[:5,:]
                df_temp["pct_"+col] = df_temp["pct_"+col].apply(lambda x : round(x*100,1))
                df_temp.index = np.arange(df_temp.shape[0])
                df2 = pd.concat([df2 , df_temp ], axis=1)
    df2.drop(["col1"],axis=1,inplace=True)
    print("those{} have more than {} occurences".format(l,max_number_occ))
    return df2

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def get_cramer(col1,col2):
    #col1 = pd.Series(col1)
    #col2 = pd.Series(col2)
    confusion_matrix = pd.crosstab(col1, col2)
    val_cramer = cramers_corrected_stat(confusion_matrix)
    return val_cramer
    
def var_ok_cramer(df,max_modalities=150):
    var_qual = df.columns[np.flatnonzero(df.dtypes=="object")]
    var_qual2 = [col for col in var_qual if df[col].nunique() <= max_modalities and df[col].nunique() > 1 ]
    print("those variables have too many modalities{}".format(set(var_qual)-set(var_qual2)))
    return var_qual2
    


def cramer_correlations(df,l_var):
    """
    df : Input a dataframe
    l_var : list of qualitative variables
    Ouput a cramer matrix
    """
    df_qual2 = df.loc[:,l_var]
    l = len(df_qual2.columns)
    results = np.zeros((l,l))
    for i, ac in enumerate(df_qual2):
        print(i),print(ac)
        for j, bc in enumerate(df_qual2):
            results[j,i] = get_cramer(df_qual2[ac],df_qual2[bc])
    results = pd.DataFrame(results,index=df_qual2.columns,columns=df_qual2.columns)
    fig, ax = plt.subplots(figsize=(21, 10))
    sns.heatmap(results, annot=True, ax=ax, cmap="YlGnBu", linewidths=0.1)
    return results
 
    

if __name__ =="main":
    df = pd.read_csv("Xy_Train_2.csv",sep=";")
    df.shape
    df.columns
    
    summary(df)
    l_var = var_ok_cramer(df,150)
    df_c = cramer_correlations(df,l_var)








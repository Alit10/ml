"""
@author = Ali 
date = 21-05-2019
"""




def summary(df):
    """"
    Takes a dataframe and return the 5 most occcurences on the object type columns
    """"
    l= []
    df2 = pd.DataFrame({"col1":np.arange(5)})
    for col in df.columns:
        if df[col].dtypes == "object":
            if df[col].nunique() > 50 :
                l.append(col)
            else : 
                df_temp = pd.DataFrame((df[col].value_counts(dropna=False)/df.shape[0]).reset_index()).rename(columns = {
        col:"pct_"+col,"index":col})
                df_temp = df_temp.sort_values(by="pct_"+col,ascending=False).iloc[:5,:]
                df_temp["pct_"+col] = df_temp["pct_"+col].apply(lambda x : int(round(x*100,0)))
                df_temp.index = np.arange(df_temp.shape[0])
                df2 = pd.concat([df2 , df_temp ], axis=1)
    df2.drop(["col1"],axis=1,inplace=True)
    print("those{} have more than 50 occurences".format(l))
    return df2
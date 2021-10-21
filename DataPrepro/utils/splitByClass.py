import os
import pandas as pd
import numpy

def split(dirName, fileType, classList, data):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    columns = data.columns.values
    dfs = {}
    for item in classList:
        dfs[item] = pd.DataFrame(columns=columns)

    for item in data.iterrows():
        index = (int)(item[1]['label'])
        val = pd.DataFrame(numpy.array(item[1][:]).reshape(1,-1), columns=columns)
        dfs[index] = dfs[index].append(val,ignore_index=True)

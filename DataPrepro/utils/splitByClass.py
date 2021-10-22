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

    #按分类读入每个df
    for item in data.iterrows():
        index = (int)(item[1]['label'])
        val = pd.DataFrame(numpy.array(item[1][:]).reshape(1,-1), columns=columns)
        dfs[index] = dfs[index].append(val,ignore_index=True)

    #把每个类别的df保存成对应的csv
    for key in dfs:
        outputName = os.path.join(dirName, fileType+'_'+str(key)+'.csv')
        dfs[key].to_csv(outputName, index=False)
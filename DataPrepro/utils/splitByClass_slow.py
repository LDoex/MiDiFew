import os
import pandas as pd
import numpy

def split(dirName, fileType, classList, data):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    total_num = data.shape[0]
    columns = data.columns.values
    dfs = {}
    for item in classList:
        dfs[item] = pd.DataFrame(columns=columns)

    print("=====start {} reading=====".format(fileType))
    #按分类读入每个df
    cur_num = 0
    for item in data.iterrows():
        if cur_num%1000==0:
            print("reading {} of total {}...".format(cur_num, total_num))
        index = (int)(item[1]['label'])
        val = pd.DataFrame(numpy.array(item[1][:]).reshape(1,-1), columns=columns)
        dfs[index] = dfs[index].append(val,ignore_index=True)
        cur_num += 1

    print("=====start saving========")
    inds = [0 for _ in range(len(dfs))]
    #把每个类别的df保存成对应的csv
    for key in dfs:
        if not os.path.isdir(os.path.join(dirName, fileType+'_'+str(key))):
            os.mkdir(os.path.join(dirName, fileType+'_'+str(key)))
        outputName = os.path.join(dirName,fileType+'_'+str(key),fileType+'_'+str(key)+'.csv')
        dfs[key].to_csv(outputName, index=False)
        inds[int(key)] = dfs[key].shape[0]
    return inds



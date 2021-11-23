import os
import pandas as pd
import numpy
import copy

def splitByNum(data, key, client_num, clients_dfs):
    num_perPart = data.size//client_num
    temp = pd.DataFrame(columns=data.column.value())
    for i in range(client_num):
        if i%num_perPart==0 or i == client_num-1:
            clients_dfs[i][key] = temp.copy(deep=True)
            temp = pd.DataFrame(columns=data.columns.values)

        temp = temp.append(data[i])

def split(dirName, fileType, classList, data, client_num):
    if not os.path.isdir(dirName):
        os.mkdir(dirName)

    columns = data.columns.values
    dfs = {}
    clients_dfs = {}
    for item in classList:
        dfs[item] = pd.DataFrame(columns=columns)

    #在每个client上深拷贝建立dfs
    for i in range(client_num):
        columns[i] = copy.deepcopy(dfs)

    #按分类读入每个df
    for item in data.iterrows():
        index = (int)(item[1]['label'])
        val = pd.DataFrame(numpy.array(item[1][:]).reshape(1,-1), columns=columns)
        dfs[index] = dfs[index].append(val,ignore_index=True)

    #把每个类别的df按client_num拆分成n份，并加入clients的dfs字典
    for key in dfs:
        splitByNum(dfs[key], key, client_num, clients_dfs)

    for key in dfs:
        outputName = os.path.join(dirName, fileType+'_'+str(key)+'.csv')
        dfs[key].to_csv(outputName, index=False)

#把单个类别的data拆分成client_num份
def split(data, client_num):

    columns = data.columns.values
    dfs = {}
    num_per_part = data.shape[0]//client_num
    for i in range(client_num):
        dfs[i] = pd.DataFrame(columns=columns)

    temp = pd.DataFrame(columns=columns)
    dfs_index = 0
    for j in range(data.shape[0]):
        print('cur_process: {} of total: {}'.format(j, data.shape[0]))
        val = pd.DataFrame(numpy.array(data.iloc[j]).reshape(1, -1), columns=columns)
        temp = temp.append(val)
        if j == data.shape[0]-1:
            if dfs_index==client_num-1:
                dfs[dfs_index] = dfs[dfs_index].append(temp)
            break
        if j!=0 and (j+1)%num_per_part==0:
            dfs[dfs_index]=dfs[dfs_index].append(temp)
            temp = pd.DataFrame(columns=columns)
            dfs_index+=1

    return dfs


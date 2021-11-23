#input: csv by class
#output: csv by train and test
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from DataPrepro.utils import splitToClient

client_num = 5
class_num = 8
class_list = [i for i in range(class_num)]
File_names = {str(i): os.path.join('./csvData/Watertank_'+str(i)+'.csv') for i in range(class_num)}
TrainOutput_paths = {str(i): os.path.join('./csvData/Watertank_client_raw_data/client{}'.format(i))
                     for i in range(client_num)}
TestOutput_paths = {str(i): os.path.join('./csvData/Watertank_client_raw_data/client{}'.format(i))
                     for i in range(client_num)}

#初始化路径文件夹
for key in TrainOutput_paths.keys():
    if not os.path.isdir(TrainOutput_paths[key]):
        os.makedirs(TrainOutput_paths[key])

DataFrames = {key: pd.read_csv(val) for key, val in File_names.items()}
Trains = {}
Tests = {}
clients = {str(i): [Trains, Tests] for i in range(client_num)}

for key, df in DataFrames.items():
    #每个df拆分成n份，对应n个client
    cur_dfs = splitToClient.split(data=df, client_num=client_num)
    #每个client的数据对应拆分成train和test,并加入到对应的client中
    for i in range(len(cur_dfs)):
        clients[str(i)][0][key], clients[str(i)][1][key] = train_test_split(cur_dfs[i], train_size=0.7, random_state=0)

for i in range(client_num):
    Train_data = pd.DataFrame(data=None)
    for key in clients[str(i)][0].keys():
        Train_data = Train_data.append(clients[str(i)][0][key])

    Test_data = pd.DataFrame(data=None)
    for key in clients[str(i)][1].keys():
        Test_data = Test_data.append(clients[str(i)][1][key])

    Train_data.to_csv(os.path.join(TrainOutput_paths[str(i)], 'Train.csv'), index=False)
    Test_data.to_csv(os.path.join(TestOutput_paths[str(i)],'Test.csv'), index=False)






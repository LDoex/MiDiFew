#input: csv by class
#output: csv by train and test
import pandas as pd
import os
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from DataPrepro.utils import splitToClient, pca_util, norm_util
import random
import time

def random_list(start,stop,length):
    if length>=0:
        length=int(length)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    random_list = []

    for i in range(length):
        cur_ind = random.randint(start, stop)
        while random_list is None and abs(cur_ind-random_list[-1])<4400 and abs(random_list[0]-cur_ind)<4400:
            cur_ind = random.randint(start, stop)
        random_list.append(cur_ind)
        random_list = sorted(random_list)

    return random_list

start_time = time.clock()
client_num = 10
class_num = 8
class_list = [i for i in range(class_num)]
#File_names = {str(i): os.path.join('./csvData/Watertank_'+str(i)+'.csv') for i in range(class_num)}

Train_file = pd.read_csv('./csvData/Watertank_Train_onehot.csv')
Test_file = pd.read_csv('./csvData/Watertank_Test_onehot.csv')

Train_sample = Train_file.sample(frac=1)
Test_sample = Test_file.sample(frac=1)

Train_shuffle = Train_sample.reset_index(drop=True)
Test_shuffle = Test_sample.reset_index(drop=True)

TrainOutput_paths = {str(i): os.path.join('./csvData/Watertank_client_data/client{}'.format(i))
                     for i in range(client_num)}
TestOutput_paths = {str(i): os.path.join('./csvData/Watertank_client_data/client{}'.format(i))
                     for i in range(client_num)}

num_perTrain = Train_shuffle.shape[0] //client_num
num_perTest = Test_shuffle.shape[0] //client_num

#初始化路径文件夹
for key in TrainOutput_paths.keys():
    if not os.path.isdir(TrainOutput_paths[key]):
        os.makedirs(TrainOutput_paths[key])

Train = []
Test = []
clients = {str(i): [Train, Test] for i in range(client_num)}
Train_inds = sorted(random_list(0, Train_shuffle.shape[0], client_num-1))
Test_inds = sorted(random_list(0, Test_shuffle.shape[0], client_num-1))
Train_inds.insert(0,0)
Train_inds.append(Train_shuffle.shape[0])
Test_inds.insert(0,0)
Test_inds.append(Test_shuffle.shape[0])

for i in range(client_num):
    print("cur_client{}".format(i))
    Train_data = Train_shuffle.iloc[Train_inds[i]:Train_inds[i+1]].reset_index(drop=True)
    Test_data = Test_shuffle.iloc[Test_inds[i]:Test_inds[i+1]].reset_index(drop=True)
    # Train_data, Test_data = pca_util.pca(Train_data, Test_data, 20)
    Train_data, Test_data = norm_util.norm(Train_data, Test_data, continiuous_num=4)
    Train_data.to_csv(os.path.join(TrainOutput_paths[str(i)], 'Train.csv'), index=False)
    Test_data.to_csv(os.path.join(TestOutput_paths[str(i)],'Test.csv'), index=False)

end_time = time.clock()
print("Running time:{}s".format(end_time - start_time))




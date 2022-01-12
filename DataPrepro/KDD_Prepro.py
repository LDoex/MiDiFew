import pandas as pd
from sklearn.preprocessing import Normalizer
from DataPrepro.utils import splitByClass
import copy
import numpy as np
from keras.utils import np_utils

Train_data = pd.read_csv('./csvData/KDDTrain+.csv')
Test_data = pd.read_csv('./csvData/KDDTest+.csv')

Train_label = Train_data.iloc[:, -1]
Train_data = Train_data.iloc[:, :-1]

Test_label = Test_data.iloc[:, -1]
Test_data = Test_data.iloc[:, :-1]


###One-hot 编码
# raw_data.drop(raw_data.columns[0], axis=1, inplace=True)#丢弃第一维特征
rest_Traindata = copy.deepcopy(Train_data)
rest_Testdata = copy.deepcopy(Test_data)
rest_Traindata = rest_Traindata.drop(rest_Traindata.columns[1:4], axis=1)#去除前三列单独处理
rest_Testdata = rest_Testdata.drop(rest_Testdata.columns[1:4], axis=1)#去除前三列单独处理
Train_protocol_col = Train_data.iloc[:,1]
Train_service_col = Train_data.iloc[:,2]
Train_flag_col = Train_data.iloc[:,3]
Test_protocol_col = Test_data.iloc[:,1]
Test_service_col = Test_data.iloc[:,2]
Test_flag_col = Test_data.iloc[:,3]
#
Train_protocol_col = np.array(Train_protocol_col)
Train_service_col = np.array(Train_service_col)
Train_flag_col = np.array(Train_flag_col)
Test_protocol_col = np.array(Test_protocol_col)
Test_service_col = np.array(Test_service_col)
Test_flag_col = np.array(Test_flag_col)
#给三种非数值型特征编码
Train_protocol_col = np_utils.to_categorical(Train_protocol_col, num_classes=3)
Train_service_col = np_utils.to_categorical(Train_service_col, num_classes=70)
Train_flag_col = np_utils.to_categorical(Train_flag_col, num_classes=11)
Test_protocol_col = np_utils.to_categorical(Test_protocol_col, num_classes=3)
Test_service_col = np_utils.to_categorical(Test_service_col, num_classes=70)
Test_flag_col = np_utils.to_categorical(Test_flag_col, num_classes=11)

Train_protocol_col = pd.DataFrame(data=Train_protocol_col, columns=['protocol_{0}'.format(i) for i in range(3)])
Train_service_col = pd.DataFrame(data=Train_service_col, columns=['service_{0}'.format(i) for i in range(70)])
Train_flag_col = pd.DataFrame(data=Train_flag_col, columns=['flag_{0}'.format(i) for i in range(11)])
Test_protocol_col = pd.DataFrame(data=Test_protocol_col, columns=['protocol_{0}'.format(i) for i in range(3)])
Test_service_col = pd.DataFrame(data=Test_service_col, columns=['service_{0}'.format(i) for i in range(70)])
Test_flag_col = pd.DataFrame(data=Test_flag_col, columns=['flag_{0}'.format(i) for i in range(11)])

one_hot_Train_data = pd.concat([Train_protocol_col, Train_service_col, Train_flag_col], axis=1)
one_hot_Test_data = pd.concat([Test_protocol_col, Test_service_col, Test_flag_col], axis=1)
columns = Train_data.columns.values

###

scaler = Normalizer().fit(rest_Traindata)
rest_Traindata = pd.DataFrame(columns=rest_Traindata.columns.values,
                              data=scaler.transform(rest_Traindata))
rest_Testdata = pd.DataFrame(columns=rest_Testdata.columns.values,
                             data=scaler.transform(rest_Testdata))
Train_data = pd.concat([rest_Traindata, one_hot_Train_data], axis=1)
Test_data = pd.concat([rest_Testdata, one_hot_Test_data], axis=1)
#处理完数据拼回原数据
Train_data['label'] = Train_label
Test_data['label'] = Test_label
Train_label_count = Train_label.unique()
Test_label_count = Test_label.unique()

Train_dir = './csvData/KDDTrainClassData_onehot/'
Test_dir = './csvData/KDDTestClassData_onehot/'
splitByClass.split(dirName=Train_dir, fileType="TrainClass",
                   classList=Train_label_count, data=Train_data)
splitByClass.split(dirName=Test_dir, fileType='TestClass',
                   classList=Test_label_count, data=Test_data)
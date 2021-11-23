import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler
from DataPrepro.utils import splitByClass, splitToClient, pca_util
from sklearn.decomposition import PCA

Train_data = pd.read_csv('./csvData/KDDTrain+.csv')
Test_data = pd.read_csv('./csvData/KDDTest+.csv')

Train_label = Train_data.iloc[:, -1]
Train_data = Train_data.iloc[:, :-1]

Test_label = Test_data.iloc[:, -1]
Test_data = Test_data.iloc[:, :-1]
columns = Train_data.columns.values

#pca前需标准化为均值为0，方差为1
scaler = StandardScaler().fit(Train_data)
Train_data = pd.DataFrame(columns=columns, data=scaler.transform(Train_data))
Test_data = pd.DataFrame(columns=columns, data=scaler.transform(Test_data))

n_comp = 10

Train_data, Test_data = pca_util.pca(Train_data, Train_label, Test_data,Test_label, n_comp)

#splitToClient
Clients_Train_dir = './csvData/KDDPcaClientData/'
Clients_Test_dir = './csvData/KDDPcaClientData/'


#splitByClass
# Train_label_count = Train_label.unique()
# Test_label_count = Test_label.unique()
# Train_dir = './csvData/KDDTrainPcaData/'
# Test_dir = './csvData/KDDTestPcaData/'
# splitByClass.split(dirName=Train_dir, fileType="TrainClass",
#                    classList=Train_label_count, data=Train_df)
# splitByClass.split(dirName=Test_dir, fileType='TestData',
#                    classList=Test_label_count, data=Test_df)

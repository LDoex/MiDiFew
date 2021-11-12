import pandas as pd
from sklearn.preprocessing import Normalizer
from DataPrepro.utils import splitByClass

Train_data = pd.read_csv('./csvData/Pipeline_Train.csv')
Test_data = pd.read_csv('./csvData/Pipeline_Test.csv')

Train_label = Train_data.iloc[:, -1]
Train_data = Train_data.iloc[:, :-1]

Test_label = Test_data.iloc[:, -1]
Test_data = Test_data.iloc[:, :-1]
#读取columns
columns = Train_data.columns.values

scaler = Normalizer().fit(Train_data)
Train_data = pd.DataFrame(columns=columns,data=scaler.transform(Train_data))
Test_data = pd.DataFrame(columns=columns,data=scaler.transform(Test_data))

#处理完数据拼回原数据
Train_data['label'] = Train_label
Test_data['label'] = Test_label
Train_label_count = Train_label.unique()
Test_label_count = Test_label.unique()

Train_dir = './csvData/PipelineTrainClassData/'
Test_dir = './csvData/PipelineTestClassData/'
splitByClass.split(dirName=Train_dir, fileType="TrainClass",
                   classList=Train_label_count, data=Train_data)
splitByClass.split(dirName=Test_dir, fileType='TestClass',
                   classList=Test_label_count, data=Test_data)
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler
from DataPrepro.utils import splitByClass, splitToClient, pca_util
from sklearn.decomposition import PCA

client_name = 0
Train_data = pd.read_csv('./csvData/KDDTrain+.csv')
Test_data = pd.read_csv('./csvData/KDDTest+.csv')

Train_label = Train_data.iloc[:, -1]

Test_label = Test_data.iloc[:, -1]

columns = Train_data.columns.values

n_comp = 20

Train_data, Test_data = pca_util.pca(Train_data, Test_data, n_comp)



Train_label_count = Train_label.unique()
Test_label_count = Test_label.unique()

Train_dir = './csvData/kddTrainClassData/'
Test_dir = './csvData/kddTestClassData/'
splitByClass.split(dirName=Train_dir, fileType="TrainClass",
                   classList=Train_label_count, data=Train_data)
splitByClass.split(dirName=Test_dir, fileType='TestClass',
                   classList=Test_label_count, data=Test_data)
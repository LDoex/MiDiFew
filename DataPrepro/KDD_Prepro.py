import pandas as pd
from sklearn.preprocessing import Normalizer

Train_data = pd.read_csv('./csvData/KDDTrain+.csv')
Test_data = pd.read_csv('./csvData/KDDTest+.csv')

Train_label = Train_data[:, -1]
Train_data = Train_data[:, :-1]

Test_label = Test_data[:, -1]
Test_data = Test_data[:, :-1]



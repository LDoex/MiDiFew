from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

def norm(Train_data, Test_data, continiuous_num):
    Train_label = Train_data.iloc[:, -1]
    Discrete_Train = Train_data.iloc[:, continiuous_num:-1]
    Continiuous_Train = Train_data.iloc[:, :continiuous_num]
    Train_data = Train_data.iloc[:, :-1]


    Test_label = Test_data.iloc[:, -1]
    Discrete_Test = Test_data.iloc[:, continiuous_num:-1]
    Continiuous_Test = Test_data.iloc[:, :continiuous_num]
    Test_data = Test_data.iloc[:, :-1]

    columns = Continiuous_Train.columns.values
    # pca前需标准化为均值为0，方差为1
    # scaler = StandardScaler().fit(Continiuous_Train)
    # Train_data = pd.DataFrame(columns=columns, data=scaler.transform(Continiuous_Train))
    # Test_data = pd.DataFrame(columns=columns, data=scaler.transform(Continiuous_Test))

    scaler = Normalizer().fit(Continiuous_Train)
    Continiuous_Train = pd.DataFrame(columns=columns, data=scaler.transform(Continiuous_Train))
    Continiuous_Test = pd.DataFrame(columns=columns, data=scaler.transform(Continiuous_Test))
    
    Train_data = pd.concat([Continiuous_Train, Discrete_Train], axis=1)
    Test_data = pd.concat([Continiuous_Test, Discrete_Test], axis=1)

    Train_data['label'] = Train_label
    Test_data['label'] = Test_label

    return Train_data, Test_data
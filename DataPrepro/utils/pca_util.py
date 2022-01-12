from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

def pca(Train_data, Test_data, n_components):
    Train_label = Train_data.iloc[:, -1]
    Train_data = Train_data.iloc[:, :-1]

    Test_label = Test_data.iloc[:, -1]
    Test_data = Test_data.iloc[:, :-1]

    # pca前需标准化为均值为0，方差为1
    scaler = StandardScaler().fit(Train_data)
    columns = Train_data.columns.values
    Train_data = pd.DataFrame(columns=columns, data=scaler.transform(Train_data))
    Test_data = pd.DataFrame(columns=columns, data=scaler.transform(Test_data))

    pca_proc = PCA(n_components=n_components)
    Train_data = pca_proc.fit_transform(Train_data)
    Test_data = pca_proc.transform(Test_data)

    labels = ['PC' + str(x) for x in range(1, n_components + 1)]
    Train_df = pd.DataFrame(Train_data, columns=labels)
    Test_df = pd.DataFrame(Test_data, columns=labels)
    Train_df['label'] = Train_label
    Test_df['label'] = Test_label

    return Train_df, Test_df
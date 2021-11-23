from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA

def pca(Train_data, Train_label, Test_data, Test_label, n_components):
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
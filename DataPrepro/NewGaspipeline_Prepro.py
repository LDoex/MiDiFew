import pandas as pd
from sklearn.preprocessing import Normalizer
from DataPrepro.utils import splitByClass
import numpy as np
import copy
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from functools import reduce
def myfillna(df, isContinuous):
    cols =df.columns.values

    # 依次处理每一列
    for col in cols:
        df_totals = pd.DataFrame(df[col].unique()).applymap(lambda x: float(x) if type(x) is str else x).to_numpy()#把str转换为float
        if isContinuous:
            df_means = np.mean(df_totals[df_totals == df_totals])  # 利用nan!=nan除去nan值
            df[col] = df[col].fillna(df_means)
        else:
            fillNum = 0.0
            while fillNum in df_totals:
                fillNum += 1
            df[col] = df[col].fillna(int(fillNum))

    return df

def One_Hot(total_df, Train_df, Test_df, field_name):
    # list of list
    total_df = np.array(total_df)
    Train_df = np.array(Train_df)
    Test_df = np.array(Test_df)

    enc = OneHotEncoder(sparse=True)
    enc.fit(total_df.reshape(-1, 1))
    Train_df = enc.transform(Train_df.reshape(-1, 1)).toarray()
    Test_df = enc.transform(Test_df.reshape(-1, 1)).toarray()
    Train_des = pd.DataFrame(data=Train_df,
                             columns=[field_name + '_{0}'.format(i) for i in range(Train_df.shape[1])])
    Test_des = pd.DataFrame(data=Test_df,
                            columns=[field_name + '_{0}'.format(i) for i in range(Test_df.shape[1])])

    # return 1代表函数非正常终止
    return copy.deepcopy(Train_des), copy.deepcopy(Test_des)

Del_cols = np.array(['binary result', 'specific result', 'categorized result']) #label name
Continious_cols = np.array(['setpoint', 'gain', 'reset rate',
                      'deadband', 'cycle time', 'rate', 'pressure measurement',
                      'crc rate', 'time'])
Discrete_cols = np.array(['address', 'function', 'length', 'system mode',
                      'control scheme', 'pump', 'solenoid', 'command response'])

Train_data = pd.read_csv('./csvData/NewGaspipeline_Train.csv')
Test_data = pd.read_csv('./csvData/NewGaspipeline_Test.csv')

Train_label = Train_data.loc[:, 'categorized result']

Test_label = Test_data.loc[:, 'categorized result']

### get continious and discrete data respectivelly
rest_Traindata = copy.deepcopy(Train_data.drop(columns=Del_cols, axis=1))
Continious_Traindata = copy.deepcopy(rest_Traindata.loc[:, Continious_cols])
Discrete_Traindata = copy.deepcopy(rest_Traindata.loc[:, Discrete_cols])

###nan值处理
Continious_Traindata = Continious_Traindata.applymap(lambda x: np.nan if '?' in str(x) else x)
Continious_Traindata = myfillna(Continious_Traindata, isContinuous=True)
Discrete_Traindata = Discrete_Traindata.applymap(lambda x: np.nan if '?' in str(x) else x)
Discrete_Traindata = myfillna(Discrete_Traindata, isContinuous=False)

rest_Testdata = copy.deepcopy(Test_data.drop(columns=Del_cols, axis=1))
Continious_Testdata = copy.deepcopy(rest_Testdata.loc[:, Continious_cols])
Discrete_Testdata = copy.deepcopy(rest_Testdata.loc[:, Discrete_cols])

###nan值处理
Continious_Testdata = Continious_Testdata.applymap(lambda x: np.nan if '?' in str(x) else x)
Continious_Testdata = myfillna(Continious_Testdata, isContinuous=True)
Discrete_Traindata = Discrete_Traindata.applymap(lambda x: np.nan if '?' in str(x) else x)
Discrete_Traindata = myfillna(Discrete_Traindata, isContinuous=False)

###one hot
full = pd.concat([Discrete_Traindata, Discrete_Testdata])

Discrete_Traindata_onehot = None
Discrete_Testdata_onehot = None
for i in Discrete_cols:
    curTrain, curTest = One_Hot(full.loc[:, i], Discrete_Traindata.loc[:, i],
                                Discrete_Testdata.loc[:, i], i)
    Discrete_Traindata_onehot = curTrain if Discrete_Traindata_onehot is None else pd.concat(
        [Discrete_Traindata_onehot, curTrain], axis=1)
    Discrete_Testdata_onehot = curTest if Discrete_Testdata_onehot is None else pd.concat(
        [Discrete_Testdata_onehot, curTest], axis=1)

###
# 读取columns
continious_columns = Continious_Traindata.columns.values

# normalization
scaler = Normalizer().fit(Continious_Traindata)
Continious_Traindata = pd.DataFrame(columns=continious_columns, data=scaler.transform(Continious_Traindata))
Continious_Testdata = pd.DataFrame(columns=continious_columns, data=scaler.transform(Continious_Testdata))
#
# # witout normalization
# Continious_Traindata = pd.DataFrame(columns=continious_columns, data=Continious_Traindata)
# Continious_Testdata = pd.DataFrame(columns=continious_columns, data=Continious_Testdata)

Train_data = pd.concat([Continious_Traindata, Discrete_Traindata_onehot], axis=1)
Test_data = pd.concat([Continious_Testdata, Discrete_Testdata_onehot], axis=1)
# 处理完数据拼回原数据
Train_data['label'] = Train_label
Test_data['label'] = Test_label
Train_label_count = Train_label.unique()
Test_label_count = Test_label.unique()

# #save the total data
# Train_data.to_csv('./csvData/NewGaspipeline_Train_onehot.csv', index=False)
# Test_data.to_csv('./csvData/NewGaspipeline_Test_onehot.csv', index=False)

# split by class
Train_dir = './csvData/NewGaspipelineTrainClassData_onehot/'
Test_dir = './csvData/NewGaspipelineTestClassData_onehot/'
splitByClass.split(dirName=Train_dir, fileType="TrainClass",
                   classList=Train_label_count, data=Train_data)
splitByClass.split(dirName=Test_dir, fileType='TestClass',
                   classList=Test_label_count, data=Test_data)
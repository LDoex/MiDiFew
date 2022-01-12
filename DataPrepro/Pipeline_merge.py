#input: csv by class
#output: csv by train and test
import pandas as pd
import os
from sklearn.model_selection import train_test_split

File_names = {str(i): os.path.join('./csvData/Pipeline_'+str(i)+'.csv') for i in range(8)}
TrainOutput_path = os.path.join('./csvData/Pipeline_Train.csv')
TestOutput_path = os.path.join('./csvData/Pipeline_Test.csv')

DataFrames = {key: pd.read_csv(val) for key, val in File_names.items()}
Trains = {}
Tests = {}

for key, df in DataFrames.items():
    Trains[key], Tests[key] = train_test_split(df, train_size=0.7, random_state=0)

Train_data = pd.DataFrame(data=None)
for key in Trains.keys():
    Train_data = Train_data.append(Trains[key])

Test_data = pd.DataFrame(data=None)
for key in Tests.keys():
    Test_data = Test_data.append(Tests[key])

Train_data.to_csv(TrainOutput_path, index=False)
Test_data.to_csv(TestOutput_path, index=False)






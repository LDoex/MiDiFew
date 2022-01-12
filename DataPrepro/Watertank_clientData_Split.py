import pandas as pd
from sklearn.preprocessing import Normalizer
from DataPrepro.utils import splitByClass
import os
import pandas as pd
import time

num_client = 10
dir_name = "./csvData/Watertank_client_data/"
total = None
start_time = time.clock()

for i in range(num_client):
    print("cur_client{}".format(i))
    cur_dir = os.path.join(dir_name,'client{}'.format(i))

    Train_data = pd.read_csv(os.path.join(cur_dir,'Train.csv'))
    Test_data = pd.read_csv(os.path.join(cur_dir,'Test.csv'))

    Train_label = Train_data.iloc[:, -1]
    Test_label = Test_data.iloc[:, -1]

    # 读取columns
    columns = Train_data.columns.values

    ###Test###
    # Train_count = [1,2,3,4,5,6,7]
    # Test_count =  [1,2,3,4,5,6,7]
    # Train_count = pd.DataFrame(Train_count)
    # Test_count = pd.DataFrame(Test_count)
    #
    #
    # cur_df = pd.concat([Train_count, Test_count], axis=1, ignore_index=True)
    # cur_df = pd.DataFrame(cur_df.values, columns=['client{}Train'.format(i), 'client{}Test'.format(i)])
    # total = cur_df.copy() if total is None else pd.concat([total, cur_df], axis=1, ignore_index=True)

    Train_label_count = Train_label.unique()
    Test_label_count = Test_label.unique()

    Train_dir = os.path.join(cur_dir,'TrainClassData')
    Test_dir = os.path.join(cur_dir,'TestClassData')
    Train_count = splitByClass.split(dirName=Train_dir, fileType="TrainClass",
                       classList=Train_label_count, data=Train_data)
    Test_count = splitByClass.split(dirName=Test_dir, fileType='TestClass',
                       classList=Test_label_count, data=Test_data)

    for item in Train_count:
        if item<40:
            print("Too small samples")
            break
    Train_count = pd.DataFrame(Train_count)
    Test_count = pd.DataFrame(Test_count)

    cur_df = pd.concat([Train_count, Test_count], axis=1, ignore_index=True)
    cur_df = pd.DataFrame(cur_df.values, columns=['client{}Train'.format(i), 'client{}Test'.format(i)])
    total = cur_df.copy() if total is None else pd.concat([total, cur_df], axis=1, ignore_index=True)

total.to_csv(os.path.join(dir_name,'count.csv'), index=False)

end_time = time.clock()
print("Running time:{}s".format(end_time-start_time))



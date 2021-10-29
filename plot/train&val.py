import os
import re
import pandas as pd
import matplotlib.pyplot as plt

filePath = "../scripts/train/midifew/results/"
fileName = "trace.txt"
df_columns = ['train_loss', 'train_acc', 'train_pre', 'train_rec', 'train_F1',
                           'val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_F1', 'epoch']
df = pd.DataFrame(columns=df_columns)

with open(os.path.join(filePath, fileName), 'r') as f:
    for line in f.readlines():
        data = line.strip('\n')
        #去掉F1避免正则匹配到其中的1
        data = re.sub("F1", "", data)
        data = re.sub("}", "", data)
        data_list = re.findall(r'\d.?\d*', data)

        data_list = list(map(float, data_list))
        df_item = pd.DataFrame(columns=df_columns, data=[data_list])
        df = df.append(df_item, ignore_index=True)

x = df['epoch'].values
y = df['train_acc'].values

plt.plot(x, y)
plt.show()
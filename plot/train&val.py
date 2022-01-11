import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

filePath = "../scripts/results/"
#filePath = "./"
fileName = "preteacher_trace.txt"
len = 11
df_columns = ['train_loss', 'train_acc', 'train_pre', 'train_rec', 'train_F1',
                           'val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_F1', 'epoch']
global_df_columns = ['val_loss', 'val_acc', 'val_pre', 'val_rec', 'val_F1']
if len==5:
    df = pd.DataFrame(columns=global_df_columns)
else:
    df = pd.DataFrame(columns=df_columns)

with open(os.path.join(filePath, fileName), 'r') as f:
    for line in f.readlines():
        data = line.strip('\n')
        #去掉F1避免正则匹配到其中的1
        data = re.sub("F1", "", data)
        data = re.sub("}", "", data)
        data_list = re.findall(r'\d+.?\d*', data)

        data_list = list(map(float, data_list))
        df_item = pd.DataFrame(columns=df.columns, data=[data_list])
        df = df.append(df_item, ignore_index=True)

x = np.array([i for i in range(0, df.shape[0])])
y = df['train_loss'].values

x_smooth = np.linspace(x.min(), x.max(), 300)#list没有min()功能调用
y_smooth = make_interp_spline(x, y)(x_smooth)
plt.plot(x_smooth, y_smooth)


print(plt.axis([-1, 50, 0.0, 0.6]))
plt.show()
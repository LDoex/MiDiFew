import numpy as np
import pandas as pd
import csv
import time
import os
import re

global label_list  # label_list为全局变量
col_names = np.array(['address', 'function', 'length',
                      'setpoint', 'gain', 'reset rate',
                      'deadband', 'cycle time', 'rate', 'system mode',
                      'control scheme', 'pump', 'solenoid', 'pressure measurement',
                      'crc rate', 'command response', 'time', 'binary result', 'categorized result',
                      'specific result'])

# 定义数据预处理函数, 按分类转换成csv
def preHandel_data():
    source_file = './rawData/NewGaspipeline.txt'
    handled_files = {str(i): os.path.join('./csvData/NewGaspipeline_'+str(i)+'.csv') for i in range(8)}
    data_files = [open(handled_file, 'w', newline='') for handled_file in handled_files.values()] # python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file, 'r') as data_source:
        csv_reader = csv.reader(data_source)
        csv_writers = {str(i): None for i in range(8)}
        for i in range(8):
            csv_writers[str(i)] = csv.writer(data_files[i])
        for key in csv_writers.keys():
            csv_writers[key].writerow(col_names)  # 打上特征行名
        count = 0  # 记录数据的行数，初始化为0
        for row in csv_reader:
            ind_name = str(re.sub(r'd','',row[-2])).strip()
            csv_writers[ind_name].writerow(row)
            count += 1
            # 输出每行数据中所修改后的状态
            print('cur_line:', count)
        for data_file in data_files:
            data_file.close()

if __name__ =='__main__':
    start_time = time.clock()
    global label_list  # 声明一个全局变量的列表并初始化为空
    label_list = []
    preHandel_data()
    end_time = time.clock()
    print("Running time:", (end_time - start_time))
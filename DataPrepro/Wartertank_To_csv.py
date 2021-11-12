import numpy as np
import pandas as pd
import csv
import time
import os
import re

global label_list  # label_list为全局变量
col_names = np.array(['command_address', 'response_address', 'command_memory',
                      'response_memory', 'command_memory_count', 'response_memory_count',
                      'comm_read_function', 'comm_write_fun', 'resp_read_fun', 'resp_write_fun',
                      'sub_function', 'command_length', 'resp_length', 'HH', 'H', 'L', 'LL',
                      'control_mode', 'control_scheme', 'pump', 'crc_rate', 'measurement', 'time',
                      'label'])

# 定义数据预处理函数, 按分类转换成csv
def preHandel_data():
    source_file = './rawData/Watertank.txt'
    handled_files = {str(i): os.path.join('./csvData/Watertank_'+str(i)+'.csv') for i in range(8)}
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
            csv_writers[str(re.sub(r'd','',row[-1])).strip()].writerow(row)
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
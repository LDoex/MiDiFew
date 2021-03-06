from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import csv

class varClass(object):
     def __init__(self):
         self.names = self.__dict__

     def setName(self, varName, fileName):
         self.names[varName] = fileName

     def getVar(self, varName):
         return self.names[varName]

name_class = 2
num_file = 6
vars = varClass()
line_types = ['b-', 'r-', 'g-.', 'g-', 'c-.', 'c-',
              'm-.', 'm-', 'y-.', 'y-', 'k-.', 'k-',]

vars.setName('re_file0', os.path.join('./', 'y_real_preteacher.csv'))
vars.setName('preProb_file0', os.path.join('./', 'decision_val_preteacher.csv'))

vars.setName('re_file1', os.path.join('./', 'y_real_global_3clients.csv'))
vars.setName('preProb_file1', os.path.join('./', 'decision_val_global_3clients.csv'))


vars.setName('re_file2', os.path.join('./', 'y_real_student0.csv'))
vars.setName('preProb_file2', os.path.join('./', 'decision_val_student0.csv'))

vars.setName('re_file3', os.path.join('./', 'y_real_client0.csv'))
vars.setName('preProb_file3', os.path.join('./', 'decision_val_client0.csv'))

vars.setName('re_file4', os.path.join('./', 'y_real_global_4clients_few.csv'))
vars.setName('preProb_file4', os.path.join('./', 'decision_val_global_4clients_few.csv'))

vars.setName('re_file5', os.path.join('./', 'y_real_student0KD.csv'))
vars.setName('preProb_file5', os.path.join('./', 'decision_val_student0KD.csv'))

vars.setName('re_file6', os.path.join('./', 'y_real_client2FT.csv'))
vars.setName('preProb_file6', os.path.join('./', 'decision_val_client2FT.csv'))

vars.setName('re_file7', os.path.join('./', 'y_real_client2.csv'))
vars.setName('preProb_file7', os.path.join('./', 'decision_val_client2.csv'))

vars.setName('re_file8', os.path.join('./', 'y_real_client3FT.csv'))
vars.setName('preProb_file8', os.path.join('./', 'decision_val_client3FT.csv'))

vars.setName('re_file9', os.path.join('./', 'y_real_client3.csv'))
vars.setName('preProb_file9', os.path.join('./', 'decision_val_client3.csv'))

vars.setName('re_file10', os.path.join('./', 'y_real_client4FT.csv'))
vars.setName('preProb_file10', os.path.join('./', 'decision_val_client4FT.csv'))

vars.setName('re_file11', os.path.join('./', 'y_real_client4.csv'))
vars.setName('preProb_file11', os.path.join('./', 'decision_val_client4.csv'))



names = ['preteacher', 'global', 'student0', 'client0', 'globalfew',  'student0kd',
         'client2FT',  'client2',  'client3FT',  'client3',  'client4FT',  'client4']

#
# re_file0 = os.path.join('./', 'y_real_client0.csv')
# preProb_file0 = os.path.join('./', 'decision_val_client0.csv')
#
# re_file1 = os.path.join('./', 'y_real_global.csv')
# preProb_file1 = os.path.join('./', 'decision_val_global.csv')
#
# re_file2 = os.path.join('./', 'y_real_client0_withoutFineTune.csv')
# preProb_file2 = os.path.join('./', 'decision_val_client0_withoutFineTune.csv')

label_dfs = []
preProb_dfs = []
for i in range(0, num_file):
    label_dfs.append(pd.read_csv(vars.getVar('re_file{}'.format(i)), header=None))
    preProb_dfs.append(pd.read_csv(vars.getVar('preProb_file{}'.format(i)), header=None))


fprs = []
tprs = []
thersholdss = []
roc_aucs = []

for i in range(0, num_file):
    fpr, tpr, thersholds = roc_curve(label_dfs[i].values.reshape(-1).tolist(),
                                     preProb_dfs[i][name_class-1].values.reshape(-1).tolist(), pos_label=name_class)
    fprs.append(fpr)
    tprs.append(tpr)
    thersholdss.append(thersholds)
    roc_aucs.append(auc(fpr, tpr))

for i in range(0,num_file):
    plt.plot(fprs[i], tprs[i], line_types[i], label=names[i]+' (area = {0:.2f})'.format(roc_aucs[i]), lw=2)


# label_df = pd.read_csv(re_file0, header=None)
# preProb_df = pd.read_csv(preProb_file0, header=None)
#
# label_df1 = pd.read_csv(re_file1, header=None)
# preProb_df1 = pd.read_csv(preProb_file1, header=None)
#
# label_df2 = pd.read_csv(re_file2, header=None)
# preProb_df2 = pd.read_csv(preProb_file2, header=None)
#
# label_df = label_df.values.reshape(-1).tolist()
# preProb_df = preProb_df[name_class-1].values.reshape(-1).tolist()
#
# label_df1 = label_df1.values.reshape(-1).tolist()
# preProb_df1 = preProb_df1[name_class-1].values.reshape(-1).tolist()
#
# label_df2 = label_df2.values.reshape(-1).tolist()
# preProb_df2 = preProb_df2[name_class-1].values.reshape(-1).tolist()
#
#
# y_label = label_df  # ??????????????????pos_label
# y_pre = preProb_df
# fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=name_class)
#
# y_label1 = label_df1 # ??????????????????pos_label
# y_pre1 = preProb_df1
# fpr1, tpr1, thersholds1 = roc_curve(y_label1, y_pre1, pos_label=name_class)
#
# y_label2 = label_df2 # ??????????????????pos_label
# y_pre2 = preProb_df2
# fpr2, tpr2, thersholds2 = roc_curve(y_label2, y_pre2, pos_label=name_class)
#
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# roc_auc = auc(fpr, tpr)
#
# roc_auc1 = auc(fpr1, tpr1)
#
# roc_auc2 = auc(fpr2, tpr2)
#
# plt.plot(fpr, tpr, 'g--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
#
# plt.plot(fpr1, tpr1, 'r--', label='ROC (area = {0:.2f})'.format(roc_auc1), lw=2)
#
# plt.plot(fpr2, tpr2, 'c--', label='ROC (area = {0:.2f})'.format(roc_auc2), lw=2)

plt.xlim([-0.05, 1.05])  # ??????x???y????????????????????????????????????????????????????????????????????????
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # ??????????????????????????????????????????????????????
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(b=True)
plt.show()
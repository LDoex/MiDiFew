from sklearn.svm import SVC # "Support vector classifier"
import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics
import torchnet as tnt
import math

train_shot = 5
test_shot = 5
times = 100

train_0 = pd.read_csv("../Data/pipeline/TrainClass_0/TrainClass_0.csv")
train_1 = pd.read_csv("../Data/pipeline/TrainClass_3/TrainClass_3.csv")
test_0 = pd.read_csv("../Data/pipeline/TestClass_0/TestClass_0.csv")
test_1 = pd.read_csv("../Data/pipeline/TestClass_3/TestClass_3.csv")

logFields = ['acc', 'precision', 'recall', 'F1']
meters = {field: tnt.meter.AverageValueMeter() for field in logFields}


for i in range(times):
    train_0_sample = train_0.sample(train_shot)
    train_1_sample = train_1.sample(train_shot)

    test_0_sample = test_0.sample(test_shot)
    test_1_sample = test_1.sample(test_shot)

    train_set = shuffle(pd.concat([train_0_sample, train_1_sample], axis=0, ignore_index=True))
    test_set = shuffle(pd.concat([test_0_sample, test_1_sample], axis=0, ignore_index=True))

    X_train = train_set.iloc[:, :-1]
    Y_train = train_set.iloc[:, -1]

    X_test = test_set.iloc[:, :-1]
    Y_test = test_set.iloc[:, -1]

    for i in range(0, len(Y_train)):
        if Y_train.iloc[i] != 0.0:
            Y_train.iloc[i] = 1.0
        if Y_test.iloc[i] != 0.0:
            Y_test.iloc[i] = 1.0

    model = SVC(kernel="linear")
    model.fit(X_train, Y_train)
    Y_pre = model.predict(X_test)

    output = {"precision": metrics.precision_score(Y_test, Y_pre),
             "recall": metrics.recall_score(Y_test, Y_pre),
             "acc": metrics.accuracy_score(Y_test, Y_pre),
             "F1": metrics.f1_score(Y_test, Y_pre)}

    for field, meter in meters.items():
        meter.add(output[field])

for field,meter in meters.items():
    mean, std = meter.value()
    print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean, 1.96 * std / math.sqrt(times)))



from sklearn.svm import SVC # "Support vector classifier"
import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics

train_shot = 5
test_shot = 5
times = 100

train_0 = pd.read_csv("../Data/kdd/TrainClass_0/TrainClass_0.csv")
train_1 = pd.read_csv("../Data/kdd/TrainClass_1/TrainClass_1.csv")
test_0 = pd.read_csv("../Data/kdd/TestClass_0/TestClass_0.csv")
test_1 = pd.read_csv("../Data/kdd/TestClass_1/TestClass_1.csv")

logFields = ['acc', 'precision', 'recall', 'F1']
metrics = {}

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

    model = SVC(kernel="linear")
    model.fit(X_train, Y_train)
    Y_pre = model.predict(X_test)

    precision = metrics.precision_score(Y_test, Y_pre)
    recall = metrics.recall_score(Y_test, Y_pre)
    accuracy = metrics.accuracy_score(Y_test, Y_pre)
    F1 = metrics.f1_score(Y_test, Y_pre)


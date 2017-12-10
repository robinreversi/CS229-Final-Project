from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

'''
train_data = pd.read_csv('../norm_train_10-1000_regular.csv')
dev_data = pd.read_csv('../norm_dev_10-1000_regular.csv')
test_data = pd.read_csv('../norm_test_10-1000_regular.csv')
'''
train_data = pd.read_csv('../train_10-1000_binary.csv')
dev_data = pd.read_csv('../dev_10-1000_binary.csv')
test_data = pd.read_csv('../test_10-1000_binary.csv')

def onehot(y):
    ret = np.zeros((len(y),12))
    for i in range(len(y)):
        num = y[i]
        ret[i,int(num)] = 1.0
    return ret

train_x = np.array(train_data.iloc[:, 1:])
train_y = onehot(np.array(train_data['0'].values))
dev_x = np.array(dev_data.iloc[:, 1:])
dev_y = np.array(dev_data['0'].values)
test_x = np.array(test_data.iloc[:, 1:])
test_y = np.array(test_data['0'].values)

classifiers = [SVC(probability=True) for _ in range(12)]
print('Training...')
for i in range(12):
    print(i)
    classifiers[i].fit(train_x,train_y[:,i])

print('Predicting...')
predict_dev = [0.0 for _ in range(dev_y.shape[0])]
for j in range(dev_y.shape[0]):
    pred = [cla.predict_proba([dev_x[j,:]])[0,1] for cla in classifiers]
    predict_dev[j] = np.argmax(pred)

print(predict_dev[:5])
print(dev_y[:5])
print(sum(predict_dev == dev_y) / float(len(dev_y)))


print('Predicting...')
predict_test = [0.0 for _ in range(test_y.shape[0])]
for j in range(test_y.shape[0]):
    pred = [cla.predict_proba([test_x[j,:]])[0,1] for cla in classifiers]
    predict_test[j] = np.argmax(pred)

print(predict_test[:5])
print(test_y[:5])
print(sum(predict_test == test_y) / float(len(test_y)))

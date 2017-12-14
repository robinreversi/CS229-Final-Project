from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np



train_data = pd.read_csv('../train_10-1000_norm.csv')
dev_data = pd.read_csv('../dev_10-1000_norm.csv')
test_data = pd.read_csv('../test_10-1000_norm.csv')
train_data = pd.concat([train_data,dev_data])
'''
train_data = pd.read_csv('../train_10-1000_binary.csv')
dev_data = pd.read_csv('../dev_10-1000_binary.csv')
test_data = pd.read_csv('../test_10-1000_binary.csv')
'''
def onehot(y):
    ret = np.zeros((len(y),12))
    for i in range(len(y)):
        num = y[i]
        ret[i,int(num)] = 1.0
    return ret


train_x = np.array(train_data.iloc[:, 1:])
train_y = onehot(np.array(train_data['0'].values))
t_y = np.array(train_data['0'].values)
dev_x = np.array(dev_data.iloc[:, 1:])
dev_y = np.array(dev_data['0'].values)
test_x = np.array(test_data.iloc[:, 1:])
test_y = np.array(test_data['0'].values)

classifiers = [SVC(probability=True, C=1.5) for _ in range(12)]
print('Training...')
for i in range(12):
    print(i)
    classifiers[i].fit(train_x,train_y[:,i])


print('Predicting dev...')
predict_dev = [0.0 for _ in range(dev_y.shape[0])]
for j in range(dev_y.shape[0]):
    pred = [cla.predict_proba([dev_x[j,:]])[0,1] for cla in classifiers]
    predict_dev[j] = np.argmax(pred)

print(predict_dev[:5])
print(dev_y[:5])
print(sum(predict_dev == dev_y) / float(len(dev_y)))

'''
confus = np.zeros((12,12))
for i in range(len(dev_y)):
    confus[int(dev_y[i]),int(predict_dev[i])] += 1
print(confus)
'''


print('Predicting test...')
predict_test = [0.0 for _ in range(test_y.shape[0])]
for j in range(test_y.shape[0]):
    pred = [cla.predict_proba([test_x[j,:]])[0,1] for cla in classifiers]
    predict_test[j] = np.argmax(pred)

print(predict_test[:5])
print(test_y[:5])
print(sum(predict_test == test_y) / float(len(test_y)))

print('Predicting train...')
predict_train = [0.0 for _ in range(t_y.shape[0])]
for j in range(t_y.shape[0]):
    pred = [cla.predict_proba([train_x[j,:]])[0,1] for cla in classifiers]
    predict_train[j] = np.argmax(pred)

print(predict_train[:5])
print(t_y[:5])
print(sum(predict_train == t_y) / float(len(t_y)))


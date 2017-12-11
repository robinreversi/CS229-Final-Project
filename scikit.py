from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
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

train_x = np.array(train_data.iloc[:, 1:])
train_y = np.array(train_data['0'].values)
dev_x = np.array(dev_data.iloc[:, 1:])
dev_y = np.array(dev_data['0'].values)
test_x = np.array(test_data.iloc[:, 1:])
test_y = np.array(test_data['0'].values)



model1 = OneVsRestClassifier(XGBClassifier())
model1.fit(train_x,train_y)
output = model1.predict(dev_x)
print(output[:5])
print(dev_y[:5])
print(sum(output == dev_y) / float(len(dev_y)))

output_t = model1.predict(test_x)
print(output_t[:5])
print(test_y[:5])
print(sum(output_t == test_y) / float(len(test_y)))

output_1 = model1.predict(train_x)
print(output_1[:5])
print(train_y[:5])
print(sum(output_1 == train_y) / float(len(train_y)))
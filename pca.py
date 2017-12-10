import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

np.set_printoptions(threshold=10000000000000)

datasets = ['train_10-1000_regular.csv', 'dev_10-1000_regular.csv', 'test_10-1000_regular.csv']

def normalize(data):
    m, n = data.shape
    means = (data.sum(axis=0) * 1.0 / m)
    data = data - means
    variance = np.square(data).sum(axis=0) * 1.0 / m
    variance[variance == 0] = 1
    data /= np.sqrt(variance)    
    return data



for filename in datasets:
    train_data = pd.read_csv(filename)
    train_x = np.array(train_data.iloc[:, 1:]).astype(float)
    train_y = np.array(train_data['0'].values).reshape((train_x.shape[0], 1))
    X_std = StandardScaler().fit_transform(train_x)
    #raw = np.array(pd.read_csv('train_data.csv')).astype(float)
    normalized = normalize(train_x)
    normalized = pd.DataFrame(np.append(train_y, normalized, axis=1))
    normalized.to_csv('norm_' + filename, index=False)
'''
print normalized

sklearn_pca = sklearnPCA(n_components = 3000)
X_pca = sklearn_pca.fit_transform(X_std)
new_data = pd.DataFrame(np.append(train_y, X_pca))
print new_data
new_data.to_csv('pca_train_data.csv', index=False)
'''


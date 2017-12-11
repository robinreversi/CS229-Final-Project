import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.set_printoptions(threshold=10000000000000)

datasets = ['dev_10-1000_regular.csv']#, 'dev_10-1000_regular.csv', 'test_10-1000_regular.csv']

def normalize(data):
    m, n = data.shape
    means = (data.sum(axis=0) * 1.0 / m)
    data = data - means
    variance = np.square(data).sum(axis=0) * 1.0 / m
    variance[variance == 0] = 1
    data /= np.sqrt(variance)    
    return data

def pca_transform(data, k):
    cov = data.T.dot(data) / float(data.shape[0])
    print cov.shape
    eig_vals, eig_vecs = np.linalg.eig(cov)
    print eig_vecs.shape
    print eig_vals.shape
    print eig_vecs[np.argsort(eig_vals), :]



for filename in datasets:
    train_data = pd.read_csv(filename)
    train_x = np.array(train_data.iloc[:, 1:]).astype(float)
    train_y = np.array(train_data['0'].values).reshape((train_x.shape[0], 1))

    normalized = normalize(train_x)
    pca_transform(normalized, 10)
    #normalized = pd.DataFrame(np.append(train_y, normalized, axis=1))
    #normalized.to_csv('norm_' + filename, index=False)


import numpy as np
import pandas as pd

np.set_printoptions(threshold=10000000000000)

def normalize(data):
	m, n = data.shape
	means = (data.sum(axis=0) * 1.0 / m)
	
	data = data - means
	variance = np.square(data).sum(axis=0) * 1.0 / m
	variance[variance == 0] = 1
	data /= np.sqrt(variance)
	
	return data


raw = np.array(pd.read_csv('train_data.csv')).astype(float)
normalized = pd.DataFrame(normalize(raw))
normalized.to_csv('normalized_train_data.csv')



from __future__ import division
"""
Ml-Knn multi label classifier for sklearn
"""
# Author: Anil Narassiguin

import numpy as np

from sklearn.neighbors import NearestNeighbors

from sklearn.datasets import make_multilabel_classification
from sklearn.cross_validation import train_test_split

class MlKNN(object):
	def __init__(self, K=10, smooth=1.0):
		self.K=K
		self.smooth=smooth

	def fit(self, X, y):
		"""
		Desc
		"""
		n_samples, n_classes = y.shape
		prior = self._compute_prior(y)
		cond0, cond1 = self._compute_cond(X, y)
		
		self.prior_ = prior
		self.cond0_ = cond0
		self.cond1_ = cond1
		self.n_classes_ = n_classes

		return self

	def predict(self, X, y, x_test, return_distance=False): # Should be same X and y as in fit
		K = self.K
		s = self.smooth	
		n_classes = self.n_classes_

		knn = NearestNeighbors(K)
		knn.fit(X)
		ids = knn.kneighbors(x_test.reshape(1, -1), return_distance=False)
		y_knn = y[ids, :][0]

		labels = []
		dists = []

		for label in range(n_classes):
			delta = sum([target[label] for target in y_knn])

			p1 = self.prior_[label]*self.cond1_[label, delta]
			p0 = self.prior_[label]*self.cond0_[label, delta]

			labels.append(int(p1 >= p0))
			dists.append(p1 / (p1+p0))

		if return_distance:
			return np.array(labels), np.array(dists)
		else:
			return np.array(labels)

	
	def _compute_prior(self, y):
		prior_prob = []
		n_samples, n_classes = y.shape
		s = self.smooth
		sum_term = np.array([target.sum() for target in y])
		prior = (s + sum_term) / (2*s + n_samples)
		return prior

	def _compute_cond(self, X, y):
		K = self.K
		s = self.smooth
		n_samples, n_classes = y.shape

		p0, p1 = [], []

		for label in range(n_classes):
			c = np.zeros(K+1)
			cn = np.zeros(K+1)
			for i, sample in enumerate(X):
				restX = np.delete(X, i, axis=0)
				restY = np.delete(y, i, axis=0)
				knn = NearestNeighbors(K)
				knn.fit(restX)
				ids = knn.kneighbors(sample.reshape(1, -1), return_distance=False)
				y_knn = restY[ids, :][0]
			
				delta = sum([target[label] for target in y_knn])
				(c if y[i, label] == 1 else cn)[delta] += 1

			p1.append((s + c) / (s*(K+1) + sum(c)))
			p0.append((s + cn) / (s*(K+1) + sum(cn)))

		return np.vstack(p0), np.vstack(p1)


if __name__ == '__main__':
	X, y = make_multilabel_classification(n_samples=1000, n_classes=50, n_labels=30)
	x_test, y_test = X[1], y[1] 
	X, y = X[1:], y[1:]

	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

	mlknn = MlKNN(K=20)
	mlknn.fit(X_train, Y_train)
	output, dists = mlknn.predict(X, y, x_test, return_distance=True)
	print output
	print y_test
	print dists

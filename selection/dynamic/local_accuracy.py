"""
OLA classifier selection
"""

# Author: Anil Narassiguin

from __future__ import division

import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import NearestNeighbors

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification

class OLA:
	"""
	Desc
	"""
	def __init__(self,
				 ensemble_clf=None,
				 knn=5,
				 X_val=None,
				 y_val=None):
		self.ensemble_clf = ensemble_clf
		self.knn = knn
		self.X_val = X_val
		self.y_val = y_val

	def select(self, x_test, metric='minkowski', p=2):
		"""
		Dynamically select classifiers in the pool relatively to a test
		pattern x_test

		Parameters
		----------
		x_test : test pattern 

		Returns
		----------
		best classifier : best classifier according to the dynamic selection scheme.
		"""
		pool = self.ensemble_clf.estimators_
		if not pool:
			raise ValueError("Fit the ensemble methiod before throwing it to \
							  the dynamic selection algorithm")

		predicted_labels = [clf.predict(x_test.reshape(1, -1)) for clf in pool]

		if len(np.unique(predicted_labels)) == 1:
			# All the classifiers agree on the predicted class
			return pool[0]
		else:
			knn = NearestNeighbors(n_neighbors=self.knn, metric=metric, p=p)
			knn.fit(self.X_val)
			iknn = knn.kneighbors(x_test.reshape(1, -1), return_distance=False)[0]
			X_knn, y_knn = X_val[iknn], y_val[iknn]

			accuracies = [accuracy_score(clf.predict(X_knn), y_knn) \
						  for clf in pool]
			i_best = np.argmax(accuracies)

			return pool[i_best]

	def predict(self, X_test):
		"""Predict class for test data
		Parameters
		----------
		X : {array-like, sparse matrix} of shape = [n_samples, n_features]

		Returns
		-------
		y : array of shape = [n_samples]
		The predicted classes.
		"""		
		y_pred = [self.select(x_test).predict(x_test.reshape(1, -1))[0] \
				  for x_test in X_test]
		return y_pred

class LCA:
	"""
	Desc
	"""
	def __init__(self,
				 ensemble_clf=None,
				 knn=5,
				 X_val=None,
				 y_val=None):
		self.ensemble_clf = ensemble_clf
		self.knn = knn
		self.X_val = X_val
		self.y_val = y_val

	def select(self, x_test, metric='minkowski', p=2):
		"""
		Dynamically select classifiers in the pool relatively to a test
		pattern x_test with the LCA scheme

		Parameters
		----------
		x_test : test pattern 

		Returns
		----------
		best classifier : best classifier according to the dynamic selection scheme.
		"""
		pool = self.ensemble_clf.estimators_
		if not pool:
			raise ValueError("Fit the ensemble methiod before throwing it to \
							  the dynamic selection algorithm")

		predicted_labels = [clf.predict(x_test.reshape(1, -1)) for clf in pool]
		if len(np.unique(predicted_labels)) == 1:
			# All the classifiers agree on the predicted class
			return pool[0]
		else:
			#knn = NearestNeighbors(n_neighbors=self.knn, metric=metric, p=p)
			#knn.fit(self.X_val)
			#iknn = knn.kneighbors(x_test.reshape(1, -1), return_distance=False)[0]
			#X_knn, y_knn = X_val[iknn], y_val[iknn]

			#accuracies = [accuracy_score(clf.predict(X_knn), y_knn) \
			#			  for clf in pool]
			#i_best = np.argmax(accuracies)
			i_best = 0
			lca = 0
			for i, clf in enumerate(pool):
				output = clf.predict(x_test.reshape(1, -1))
				X_lca = self.X_val[y_val == output]

				knn = NearestNeighbors(n_neighbors=self.knn, metric=metric, p=p)
				knn.fit(X_lca)
				iknn = knn.kneighbors(x_test.reshape(1, -1), return_distance=False)[0]
				X_knn = X_val[iknn]

				accuracy = sum(clf.predict(X_knn) == output) / len(output)

				if accuracy > lca:
					i_best = i
					lca = accuracy

			return pool[i_best]

	def predict(self, X_test):
		"""Predict class for test data
		Parameters
		----------
		X : {array-like, sparse matrix} of shape = [n_samples, n_features]

		Returns
		-------
		y : array of shape = [n_samples]
		The predicted classes.
		"""		
		y_pred = [self.select(x_test).predict(x_test.reshape(1, -1))[0] \
				  for x_test in X_test]
		return y_pred

		

if __name__ == '__main__':
	X, y = make_classification(n_samples=1000)
	x_test, y_test = X[0], y[0]
	X, y = X[1:], y[1:]


	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train \
	                                   , test_size=0.2)
	
	bag = BaggingClassifier(n_estimators=50)
	bag.fit(X_train, y_train)

	ola = OLA(bag, 6, X_val, y_val)
	lca = LCA(bag, 6, X_val, y_val)
	
	acc_ola = []
	acc_lca = []
	acc_bag = []
	for it in range(50):
		if not it % 5:
			print "Iteration number %s :" % it
		
		acc_lca.append(accuracy_score(lca.predict(X_test), y_test))
		acc_ola.append(accuracy_score(ola.predict(X_test), y_test))		
		acc_bag.append(accuracy_score(bag.predict(X_test), y_test))

	print "ACC OLA", np.mean(acc_ola)
	print "ACC LCA", np.mean(acc_lca)
	print "ACC BAG", np.mean(acc_bag)






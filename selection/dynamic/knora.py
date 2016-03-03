"""
KNORA ensemble selection
"""

# Author: Anil Narassiguin

from __future__ import division

import numpy as np

from collections import defaultdict
import operator

from abc import ABCMeta

from sklearn.externals.six import with_metaclass
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.base import ClassifierMixin

from base import BaseDynamic

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification

class KNORA(BaseDynamic, ClassifierMixin):
	"""
	Desc
	"""
	def __init__(self,
				 base_estimator=DecisionTreeClassifier(),
				 n_estimators=50,		
				 ensemble_clf=None,
				 knn=5,
				 metric='minkowski',
				 p=2,
				 X_val=None,
				 y_val=None,
				 scheme='ELIMINATE'):
		self.knn = knn
		self.scheme = scheme
		self.metric = metric
		self.p = p
		super(KNORA, self).__init__(
			base_estimator=base_estimator,
			n_estimators=n_estimators,
			ensemble_clf=ensemble_clf,
			X_val=X_val,
			y_val=y_val)

			

	#def fit(X, y):
	#	# define it once the class is finished
	#	pass

	#def predict(self, X, metric='minkowski', p=2):
	#	scheme = self.scheme
	#	if scheme == 'ELIMINATE':
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
		y_pred = [self.predict_pattern(x_test) for x_test in X_test]
		return y_pred

	def predict_pattern(self, x_test):
		best_pool, weights = self.select(x_test)
		nb_clf = len(best_pool)
		results = defaultdict(float)
		if weights is None:
			weights = np.ones(nb_clf)
			weights /= sum(weights)

		for i, clf in enumerate(best_pool):
			y_pred = clf.predict(x_test.reshape(1, -1))
			results[y_pred[0]] += weights[i]

		return max(results.iteritems(), key=operator.itemgetter(1))[0]

	def select(self, x_test):
		scheme = self.scheme
		if scheme.startswith('ELIMINATE'):
			return self.eliminate(x_test)
		elif scheme.startswith('UNION'):
			return self.union(x_test)		

	def eliminate(self, x_test):
		"""
		Dynamically select classifiers in the pool relatively to a test
		pattern x_test and using the KNORA-ELIMINATE scheme

		Parameters
		----------
		x_test : test pattern 
		weighted : 

		Returns
		----------
		best classifiers : best pool of classifiers according to the dynamic selection scheme.
		"""	
		K=self.knn
		metric=self.metric
		p=self.p
		keep_running=True
		pool = self.ensemble_clf.estimators_
		scheme = self.scheme

		while keep_running:
			best_pool = []

			# Finding the neighbors
			knn = NearestNeighbors(n_neighbors=K, metric=metric, p=p)
			knn.fit(self.X_val)
			weights, iknn = knn.kneighbors(x_test.reshape(1, -1))
			weights = weights[0]
			weights /= sum(weights)
			iknn = iknn[0]
			X_knn, y_knn = self.X_val[iknn], self.y_val[iknn]

			K -= 1			
			
			for clf in pool:
				accuracy = accuracy_score(y_knn, clf.predict(X_knn))

				if accuracy == 1.:
					keep_running = False
					best_pool.append(clf)

			# How to handle the case where no classifier is correct on knn data ??
			if K == 0 and len(best_pool) == 0:
				best_pool = np.copy(pool)
				keep_running = False
		
		if scheme.endswith("W"):
			return best_pool, weights
		else:
			return best_pool, None

	def union(self, x_test):
		"""
		Dynamically select classifiers in the pool relatively to a test
		pattern x_test and using the KNORA-UNION scheme

		Parameters
		----------
		x_test : test pattern 
		weighted : 

		Returns
		----------
		best classifiers : best pool of classifiers according to the dynamic selection scheme.
		"""
		K = self.knn
		metric=self.metric
		p=self.p			
		pool = self.ensemble_clf.estimators_	
		best_pool = []
		scheme = self.scheme

		# Finding the neighbors
		knn = NearestNeighbors(n_neighbors=K, metric=metric, p=p)
		knn.fit(self.X_val)
		weights, iknn = knn.kneighbors(x_test.reshape(1, -1))
		weights = weights[0]
		weights /= sum(weights)
		iknn = iknn[0]
		X_knn, y_knn = self.X_val[iknn], self.y_val[iknn]

		for clf in pool:
			for i in range(X_knn.shape[0]):
				if clf.predict(X_knn[i,:].reshape(1, -1)) == y_knn[i]:
					best_pool.append(clf)
			
		if scheme.endswith("W"):
			return best_pool, weights
		else:
			return best_pool, None

if __name__ == '__main__':
	X, y = make_classification(n_samples=1000)
	x_test, y_test = X[0], y[0]
	X, y = X[1:], y[1:]


	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train \
	                                   , test_size=0.2)
	
	bag = BaggingClassifier(n_estimators=50)
	bag.fit(X_train, y_train)
	knora = KNORA(ensemble_clf=bag, knn=8, X_val=X_val, y_val=y_val)

	print accuracy_score(bag.predict(X_test), y_test)
	print accuracy_score(knora.predict(X_test), y_test)

	#best_pool = knora.union(x_test)
	#print best_pool
	#print len(best_pool)
















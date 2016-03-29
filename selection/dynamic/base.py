"""
Base class for dynamic ensemble selection
"""

# Author: Anil Narassiguin

from __future__ import division

import numpy as np

from abc import ABCMeta, abstractmethod

from sklearn.externals.six import with_metaclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble.base import BaseEnsemble

from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification

class BaseDynamic(with_metaclass(ABCMeta, BaseEnsemble)):
	"""
	Desc
	"""
	@abstractmethod
	def __init__(self,
				 base_estimator=DecisionTreeClassifier(),
				 n_estimators=50,
				 ensemble_clf=None,
				 X_val=None,
				 y_val=None):
		self.ensemble_clf = ensemble_clf
		if hasattr(self.ensemble_clf, 'n_classes_'):
			self.n_classes_ = self.ensemble_clf.n_classes_
		self.X_val = X_val
		self.y_val = y_val 
		super(BaseDynamic, self).__init__(
			base_estimator=base_estimator,
			n_estimators=n_estimators)

	def fit(self, X, y, sample_weight=None):
		n_estimators = self.n_estimators
		base_estimator = self.base_estimator 
		clf = self.ensemble_clf
		clf.set_params(**{'n_estimators': n_estimators, 'base_estimator': base_estimator})
		clf.fit(X, y, sample_weight=sample_weight)
		self.ensemble_clf = clf
		self.classes_ = clf.classes_
		self.n_classes_ = clf.n_classes_
		return self

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
		y_pred = []
		pool_sizes = []
		for x_test in X_test:
			target, pool_size = self.predict_pattern(x_test)
			y_pred.append(target)
			pool_sizes.append(pool_size)

		self.pool_sizes_ = pool_sizes

		#y_pred = [self.predict_pattern(x_test) for x_test in X_test]

		return np.array(y_pred)

	def predict_proba(self, X_test):
		# Voting
		n_samples = X_test.shape[0]
		proba = np.zeros((n_samples, self.n_classes_))
		pool_sizes = np.zeros(n_samples)

		for i, x_test in enumerate(X_test):
			best_pool = self.select(x_test)
			pool_size = len(best_pool)
			pool_sizes[i] = pool_size
			for estimator in best_pool:
				target = estimator.predict(x_test.reshape(1, -1))
				proba[i, target] += 1

		self.pool_sizes_ = pool_sizes

		return proba / pool_sizes[:, np.newaxis]

	@abstractmethod
	def select(self, x_test):
		pass

	@abstractmethod
	def predict_pattern(self, x_test):
		pass









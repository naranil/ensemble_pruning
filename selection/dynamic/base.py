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
		return self

	@abstractmethod
	def select(self, x_test):
		pass










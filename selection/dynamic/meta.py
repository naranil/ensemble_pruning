"""
META DES algorithms:
IBEP_MLC from Markatopoulou
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin

from base import BaseDynamic
from knora import KNORA

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification

class IbepMlc(BaseDynamic, ClassifierMixin):
	"""
	Desc
	"""
	def __init__(self,
				 base_estimator=DecisionTreeClassifier(),
				 n_estimators=50,		
				 ensemble_clf=None,
				 knn=10,
				 X_val=None,
				 y_val=None):
		self.knn = knn
		super(IbepMlc, self).__init__(
			base_estimator=base_estimator,
			n_estimators=n_estimators,
			ensemble_clf=ensemble_clf,
			X_val=X_val,
			y_val=y_val)

	def mlknn(self):
		pool = self.ensemble_clf.estimators_
		K = self.knn
		X_val = self.X_val
		y_val = self.y_val
		
		ml_knn = KNeighborsClassifier(n_neighbors=K)
		meta_y = np.vstack([clf.predict(X_val) == y_val for clf in pool])
		#print X_val.shape
		#print meta_y.shape

		ml_knn.fit(X_val, meta_y.T)
		return ml_knn

	def select(self, x_test):
		pool = np.array(self.ensemble_clf.estimators_)
		ml_knn = self.mlknn()
		best_pool_idx = ml_knn.predict(x_test.reshape(1, -1))
		best_pool = pool[best_pool_idx.astype(bool)[0]]
		print "Best pool size %s" % len(best_pool)
		return best_pool

	def predict(self, X_test):
		"""Desc
		"""		
		y_pred = []
		for x_test in X_test:
			best_pool = self.select(x_test)
			for clf in best_pool:
				results = defaultdict(float)
				y = clf.predict(x_test.reshape(1, -1))
				results[y[0]] += 1

			y = max(results.iteritems(), key=operator.itemgetter(1))[0]
			y_pred.append(y)

		return y_pred

if __name__ == '__main__':
	X, y = make_classification(n_samples=1000, n_features=20, class_sep=0.7, flip_y=0.03)
	x_test, y_test = X[0], y[0]
	X, y = X[1:], y[1:]


	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train \
	                                   , test_size=0.2)
	
	bag = BaggingClassifier(n_estimators=200)
	bag.fit(X_train, y_train)
	knora = KNORA(ensemble_clf=bag, knn=8, X_val=X_val, y_val=y_val)
	meta = IbepMlc(ensemble_clf=bag, knn=8, X_val=X_val, y_val=y_val)

	print accuracy_score(bag.predict(X_test), y_test)
	print accuracy_score(knora.predict(X_test), y_test)
	print accuracy_score(meta.predict(X_test), y_test)



"""
Probabilistic model of classifier competence for dynamic ensemble selection
"""
# Author: Anil Narassiguin

from __future__ import division

import numpy as np
from numpy.matlib import repmat
from numpy.linalg import norm

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

from scipy.special import betainc

class DesCV(BaseDynamic):
	"""
	Desc
	"""
	def __init__(self,
				 base_estimator=DecisionTreeClassifier(),
				 n_estimators=50,		
				 ensemble_clf=None,
				 X_val=None,
				 y_val=None):
		super(DesCV, self).__init__(
			base_estimator=base_estimator,
			n_estimators=n_estimators,
			ensemble_clf=ensemble_clf,
			X_val=X_val,
			y_val=y_val)

	def predict_pattern(self, x_test):
		best_pool = self.select(x_test)
		print len(best_pool)
		results = defaultdict(float)
		for estimator in best_pool:
			y_pred = estimator.predict(x_test.reshape(1, -1))
			results[y_pred[0]] += 1

		return max(results.iteritems(), key=operator.itemgetter(1))[0]

	def select(self, x_test):
		n_classes = self.ensemble_clf.n_classes_
		estimators = self.ensemble_clf.estimators_

		scores = self.scoreDesCV(x_test)

		sort_scores = np.sort(scores)[::-1]
		sort_idx = np.argsort(scores)[::-1]
		idx = sort_idx[sort_scores > 1. / n_classes]

		return [estimators[i] for i in idx]

	def scoreDesCV(self, x_test):
		competences = self._competences()
		X_val = self.X_val
		y_val = self.y_val
		competences_test = np.zeros(competences.shape[1])

		normalisation_distance=0
		for j in range(X_val.shape[0]):
			distance = np.exp(-norm(x_test - X_val[j, :])**2)
			normalisation_distance += distance
			competences_test += competences[j, :]*distance
		competences_test /= normalisation_distance

		return competences_test

	def _competences(self):
		n_classes = self.ensemble_clf.n_classes_
		classes = self.ensemble_clf.classes_
		estimators = self.ensemble_clf.estimators_
		X_val = self.X_val
		y_val = self.y_val
		nb_estimators = len(estimators)

		competences = np.zeros((len(y_val), nb_estimators))
		for l, estimator in enumerate(estimators):
			probas = estimator.predict_proba(X_val)
			if probas.shape[1] != n_classes:
				temp = np.copy(probas)
				probas = np.zeros((temp.shape[1], n_classes))
				for i in range(temp.shape[1]):
					temp[:, classes[i]] = temp[:, i]

			competences[:, l] = self._ccprmod(probas, y_val ).ravel()

		return competences


	def _ccprmod(self, proba, target, B=20):
		"""
		Cf MATLAB code
		CCPRMOD Classifier competence based on probabilistic modelling
		
		cc = ccprmod(d,j,B)
		
		Input:
		proba - NxC matrix of normalised C class supports produced by the classifier for N objects
		target - Nx1 vector of indices of the correct classes for N objects
		B - number of points used in the calculation of the competence, higher values result in a more accurate estimation (optional, default B=20)
		
		Output:
		competences - Nx1 vector of the classifier competences		
		"""
		n_sample, n_classes = proba.shape

		# Generating points
		x = np.linspace(0, 1, B)
		x = repmat(x, n_sample, n_classes)
		
		# Calculating parameters of the beta pdfs
		a = np.zeros(x.shape)
		b = np.zeros(x.shape)
		betaincj = np.zeros(x.shape)

		for c in range(n_classes):
			a[:, c*B: (c+1)*B] = repmat(n_classes*proba[:,c].reshape(-1, 1),1,B)

		b = n_classes - a
		a[a==0] = 1e-9
		b[b==0] = 1e-9
		x[x==0] = 1e-9
		betaincj = betainc(a, b, x)

		# calculating competences
		cc = np.zeros((n_sample, 1))
		for n in range(n_sample):
			t = range(target[n]*B, (target[n]+1)*B)
			bc = betaincj[n, t]
			setdiff = list(set(range(n_classes*B)) - set(t))
			bi = betaincj[n, setdiff]
			bi = np.reshape(bi, (n_classes-1, B))
			cc[n] = sum((bc[1:] - bc[:-1])*np.prod((bi[:,:-1] + bi[:,1:])/2, axis=0))

		return cc

if __name__ == '__main__':
    X, y = make_classification(n_samples=300)
    x_test, y_test = X[0], y[0]
    X, y = X[1:], y[1:]


    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train \
                                       , test_size=0.2)


    bag = BaggingClassifier(n_estimators=30)
    bag.fit(X_train, Y_train)
    Y_bag = bag.predict(X_test)

    desCV = DesCV(ensemble_clf=bag, X_val=X_val, y_val=Y_val)
    #y_pred = desCV.predict_pattern(x_test)

    #print y_pred
    #print y_test
    Y_pred = desCV.predict(X_test)
    print Y_pred
    print Y_test

    print accuracy_score(Y_pred, Y_test)
    print accuracy_score(Y_bag, Y_test)

















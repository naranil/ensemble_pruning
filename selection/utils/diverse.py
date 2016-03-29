from __future__ import division
"""
Diverse functions used in our study
"""

import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

def rms_score(y_true, y_prob):
	# 1-RMS score function
    return 1 - np.sqrt(np.average((y_true-y_prob)**2))

def output_profile(pool, X, y):
    """The output profile of a pool of classifier on a certain dataset (X, y)
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape = [n_samples, n_features]
        The training input samples. Sparse matrices are accepted only if
        they are supported by the base estimator.
    y : array-like, shape = [n_samples]
        The target values (class labels in classification, real numbers in
        regression).
    Returns
    -------
    Output profile: For each sample in (X, y) corresponds a binary vector: 1 (0) means that
    the estimator gave a right (wrong) prediction on the sample.
    """	
    output = [[int(estimator.predict(X[i,:].reshape(1, -1))==y[i]) for estimator in pool] \
               for i in range(len(X))]

    return np.array(output)

def compare_path(path1, path2):
	#path_min = np.copy(path1) if len(path1) < len(path2) else np.copy(path2)
	#path_max = np.copy(path1) if len(path1) > len(path2) else np.copy(path2)
	intersection = np.intersect1d(path1, path2)
	return len(intersection) / len(path1)





def tree_path(node, left, right):
    """The path in a tree to arrive to a certain node
    Parameters
    ----------
    node : node in a tree
    left : children left
    right : children right
    Returns
    -------
    Path in the tree to reach "node".
    """
    final_node = node
    path = [node]
    while final_node != 0:
    	if final_node in left:
    		parent = np.where(left == final_node)[0][0]
    		path.append(parent)
    		final_node = parent
    	else:
    		parent = np.where(right == final_node)[0][0]
    		path.append(parent)
    		final_node = parent

    return np.array(path)[::-1]    		


if __name__=='__main__':

	test_output = False
	if test_output:
		X, Y = make_classification(n_samples=1200, n_features=30)
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
		bag = BaggingClassifier(n_estimators=200)
		bag.fit(X_train, Y_train)
		pool = bag.estimators_
		output = output_profile(pool, X_test, Y_test)

	test_tree_path = True
	if test_tree_path:
		children_left = np.array([ 1, -1,  3,  4,  5, -1, -1,  8, -1, 10, -1, \
			                     -1, 13, 14, -1, -1, -1])
		children_right = np.array([ 2, -1, 12, 7,  6, -1, -1,  9, -1, 11, -1, \
			                     -1, 16, 15, -1, -1, -1])
		
		leaf = 10

		print tree_path(leaf, children_left, children_right)

		print compare_path(np.array([0, 2, 3]), np.array([0, 2, 3, 7, 9, 11]))


















	

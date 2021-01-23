 ##logistic regression multi-label classification

import scipy.io as sio
import numpy 
import sklearn
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
global returnprecsion
from sklearn.model_selection import KFold
from multilabelClassificationMetrics import metrics_precision_recall
from sklearn.metrics import accuracy_score
import statistics




def build_word_vector_matrix(vector_file, n_words):
	'''Return the vectors and labels for the first n_words in vector file'''
	numpy_arrays = []
	labels_array = []
	with open(vector_file, 'r') as f:
		for c, r in enumerate(f):
			sr = r.split()

			labels_array.append(sr[0])
			numpy_arrays.append( numpy.array([float(i) for i in sr[1:]]) )
			if c == n_words :
				return numpy.array( numpy_arrays[1:] )
	return numpy.array( numpy_arrays )




def averageprec(x, Y):
	random_state = numpy.random.RandomState(0)
	n_classes = Y.shape[1]
	X = preprocessing.scale(x)


	# Run classifier
	classifier = OneVsRestClassifier(sklearn.linear_model.LogisticRegression(random_state=random_state, max_iter = 3000))

	#Kfold cross-validation

	kf = KFold(n_splits=10, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator


	accuracy = []
	returnprecision = []


	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		classifier.fit(X_train, Y_train)
		preds = classifier.predict(X_test)
		accuracy.append(accuracy_score(Y_test, preds))


	accuracy_mean = statistics.mean(accuracy)
	accuracy_std = statistics.stdev(accuracy)
	print('accuracy of logistic regression is:', accuracy_mean, accuracy_std)


	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5, random_state=random_state)
	# We use OneVsRestClassifier for multi-label prediction
	classifier.fit(X_train, Y_train)
	y_score = classifier.decision_function(X_test)
	metrics_precision_recall(Y_test, y_score, X_test, n_classes, 'Logistic Regression')
 

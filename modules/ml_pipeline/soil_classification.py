#soil_classification.py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import linear_model
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
import seaborn as sns
import multilabelClassification as mc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from multilabelClassificationMetrics import metrics_precision_recall
import NN as nn_nn
import matplotlib as mpl
import statistics
from sklearn.decomposition import PCA, IncrementalPCA, NMF

sns.set_style("white")
sns.set_palette("Set2")
plt.style.use('seaborn-white')


def logistic_regression(X, y):
	print('logistic regression')
	print(mc.averageprec(X, y))



def xgboost(X, y):
	datamatrix  = xgb.DMatrix(data=X, label=y)
	n_classes = y.shape[1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 123)
	xg_reg = OneVsRestClassifier(xgb.XGBClassifier())
	params = {"objective": 'multi:softprob', "eval_metric": 'merror','colsample_bytree' : 0.3, "learning_rate" : 0.1 , "max_depth" : 5, "alpha" : 10, "n_estimators" :10, "num_class" :y.shape[1]}
	xg_reg.fit(X_train, y_train)
	preds = xg_reg.predict(X_test)
	y_score = xg_reg.predict_proba(X_test)
	accuracy = accuracy_score(y_test, preds)
	print("accuracy Xgboost: %f", (accuracy))
	metrics_precision_recall(y_test, preds, X_test, n_classes, 'XGboost')
	#cv_results = xgb.cv(dtrain=datamatrix, params=params, nfold=10,
	#               num_boost_round=50,early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
	#print(cv_results)
	#print((cv_results["test-rmse-mean"]).tail(1))
	xg_reg = xgb.train(params=params, dtrain=datamatrix, num_boost_round=10)
	xgb.plot_tree(xg_reg,num_trees=0)
	plt.rcParams['figure.figsize'] = [70, 70]
	plt.rcParams.update({'font.size': 6})
	plt.savefig('tree.png')
	plt.close()
	#plt.show()
	plt.rcParams.update({'font.size': 65})
	xgb.plot_importance(xg_reg, max_num_features=10)
	plt.rcParams['figure.figsize'] = [5, 5]
	plt.savefig('xgboost')
	plt.close()



def svmclassifier(X, Y):
	scaler = StandardScaler().fit(X)
	X = scaler.transform(X)
	classifier = LinearSVC(random_state=0, tol=1e-3)
	#Kfold cross-validation
	n_classes = 4
	kf = KFold(n_splits=10, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	print(kf) 
	accuracy = []
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		classifier.fit(X_train, Y_train)
		y_score = classifier.decision_function(X_test)
		preds = classifier.predict(X_test)
		accuracy.append(accuracy_score(Y_test, preds))

	accuracy_mean = statistics.mean(accuracy)
	accuracy_std = statistics.stdev(accuracy)
	print('svm:', accuracy_mean, accuracy_std)
	#import pdb
	#pdb.set_trace()
	classifier = LinearSVC(random_state=0, tol=1e-5)
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2,  random_state = 123)
	classifier.fit(X_train, y_train)
	y_score = classifier.decision_function(X_test)
	metrics_precision_recall(onehotencoder(y_test.reshape(-1,1)), y_score, X_test, n_classes, 'SVM')
		
	

def replaceZeroes(data):
  min_nonzero = np.min(data[np.nonzero(data)])
  data[data == 0] = min_nonzero
  return data

def neauralnetwork_classification(X, y):
	print('neural network')
	nn_nn.classification(X, y) 

def dimensionality_reduction_f_classify(X, y):
	# Split dataset to select feature and evaluate the classifier
	X_train, X_test, y_train, y_test = train_test_split(
			X, y, stratify=y, random_state=0
	)

	plt.figure(1)
	plt.rcParams['figure.figsize'] = [50, 50]
	plt.rcParams.update({'font.size': 6})
	plt.clf()
	X_indices = np.arange(X.shape[-1])
	# #############################################################################
	# Univariate feature selection with F-test for feature scoring
	# We use the default selection function to select the four
	# most significant features
	selector = SelectKBest(f_classif, k=5)
	selector.fit(X_train, y_train)
	scores = -np.log10(replaceZeroes(selector.pvalues_))
	scores /= scores.max()
	#print(scores)
	plt.rcParams['figure.figsize'] = [70, 70]
	plt.bar(X_indices - .45, scores, width=.2,
			label=r'Univariate score ($-Log(p_{value})$)',
			edgecolor='black')

	# #############################################################################
	# Compare to the weights of an SVM
	clf = make_pipeline(MinMaxScaler(), LinearSVC())
	clf.fit(X_train, y_train)
	print('Classification accuracy without selecting features: {:.3f}'
		  .format(clf.score(X_test, y_test)))

	svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
	svm_weights /= svm_weights.sum()

	plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
			 edgecolor='black')

	clf_selected = make_pipeline(
			SelectKBest(f_classif, k=5), MinMaxScaler(), LinearSVC()
	)
	clf_selected.fit(X_train, y_train)
	print('Classification accuracy after univariate feature selection: {:.3f}'
		  .format(clf_selected.score(X_test, y_test)))

	svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
	svm_weights_selected /= svm_weights_selected.sum()

	plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
			width=.2, label='SVM weights after selection',
			edgecolor='black')

	plt.title("Comparing feature selection")
	plt.xlabel('Feature number')
	plt.yticks(())
	#plt.axis('tight')
	#plt.legend(loc='upper right')
	
	plt.legend(loc='upper left', prop={'size': 8})
	plt.savefig('dim_reduction_f_classify')
	plt.close()



def compare_methods(X, y):
	pipe = Pipeline([
	# the reduce_dim stage is populated by the param_grid
	('reduce_dim', 'passthrough'),
	('classify', LinearSVC(dual=False, max_iter=10000))
	])

	N_FEATURES_OPTIONS = [2, 4, 8]
	C_OPTIONS = [1, 10, 100, 1000]
	param_grid = [
		{
			'reduce_dim': [PCA(iterated_power=7), NMF()],
			'reduce_dim__n_components': N_FEATURES_OPTIONS,
			'classify__C': C_OPTIONS
		},
		{
			'reduce_dim': [SelectKBest(chi2)],
			'reduce_dim__k': N_FEATURES_OPTIONS,
			'classify__C': C_OPTIONS
		},
	]
	reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']

	grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)
	grid.fit(X, y)
	mean_scores = np.array(grid.cv_results_['mean_test_score'])
	# scores are in the order of param_grid iteration, which is alphabetical
	mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
	# select score for best C
	mean_scores = mean_scores.max(axis=0)
	bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
				   (len(reducer_labels) + 1) + .5)
	plt.figure()
	#plt.rcParams['figure.figsize'] = [100, 100]
	plt.rcParams.update({'font.size': 25})
	plt.rcParams['figure.figsize'] = [50, 50]
	#plt.rcParams.update({'font.size': 10})
	#plot.legend(loc=2, prop={'size': 6})
	COLORS = 'bgrcmyk'
	for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
		plt.bar(bar_offsets + i, reducer_scores, label=label)

	plt.title("Comparing feature reduction techniques")
	plt.xlabel('Reduced number of features')
	plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
	plt.ylabel('classification accuracy')
	plt.ylim((0, 1))
	plt.legend(loc='upper left', prop={'size': 20})
	plt.savefig('compare_methods')
	plt.close()

def encoded_labels(df, response_columns):
	response_column = df[response_columns]
	label_encoder = LabelEncoder()
	label_encoder = label_encoder.fit(response_column)
	label_encoded_y = label_encoder.transform(response_column)

	df.drop(response_columns, axis = 1, inplace = True)
	# Put whatever series you want in its place
	df['TEXTURE'] = label_encoded_y
	return df

def onehot_encoder(df, response_columns):
	df = encoded_labels(df, response_columns)
	a = np.array(df[response_columns].to_numpy().reshape(1, df[response_columns].to_numpy().shape[0]))
	b = np.zeros((a.size, a.max()+1))
	b[np.arange(a.size),a] = 1
	return b

def onehotencoder(X):
	ohe = OneHotEncoder() 
	X_ohe = ohe.fit_transform(X) # It returns an numpy array
	return X_ohe.toarray()

def compare_methods_dim_reduction(df, columns, response_columns):
	dimensionality_reduction_f_classify(df[columns].to_numpy() , df[response_columns].to_numpy())
	compare_methods(df[columns].to_numpy() , df[response_columns].to_numpy())


def classification(df, columns, response_columns):
	b = onehot_encoder(df, response_columns)
	# CLASSIFICATION
	neauralnetwork_classification(df[columns].to_numpy() ,b)
	logistic_regression(df[columns] , b)
	svmclassifier(df[columns].to_numpy() ,df[response_columns].to_numpy())
	xgboost(df[columns] , b)
	compare_methods_dim_reduction(df, columns, response_columns)

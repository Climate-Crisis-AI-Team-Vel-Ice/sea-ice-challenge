#soil_prediction.py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import linear_model
import sklearn
from sklearn.svm import LinearSVC
from sklearn import linear_model
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
import multilabelClassification as mc
from sklearn.preprocessing import MinMaxScaler
import NN as nn_nn
import NN_regression as nr
import matplotlib as mpl
from sklearn.isotonic import IsotonicRegression
import statistics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
import NN_regression_pytorch as nrp
import multioutput_nn_regressor as mnr
from sklearn.multioutput import RegressorChain
from sklearn.multioutput import MultiOutputRegressor


sns.set_style("white")
sns.set_palette("Set2")
plt.style.use('seaborn-white')



def Linear_Regression(X, Y):
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# Create linear regression object
		regr = linear_model.LinearRegression()

		# Train the model using the training sets
		regr.fit(X_train, y_train)

		# Make predictions using the testing set
		y_pred = regr.predict(X_test)

		# The coefficients
		#print('Coefficients: \n', regr.coef_)
		coefficients = coefficients + regr.coef_

		# The mean squared error
		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y_test, y_pred))
		r2score.append(r2_score(y_test, y_pred))

	print('Coefficients: \n', np.multiply(coefficients,0.2))
	# The mean squared error
	print('Mean squared error: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.2f'
			  % statistics.mean(r2score))



def svreg(X, Y):
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# Create Decision Tree regression object
		   # Fit regression model
		regr= SVR(C=1.0, epsilon=0.2)

		# Train the model using the training sets
		regr.fit(X_train, y_train)

		# Make predictions using the testing set
		y_pred = regr.predict(X_test)

		# The mean squared error

		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y_test, y_pred))
		r2score.append(r2_score(y_test, y_pred))

	# The mean squared error
	print('Mean squared error SVR: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination SVR: %.2f'
			  % statistics.mean(r2score))

def Knnn(X, Y):
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# Create Decision Tree regression object
		   # Fit regression model
		regr= KNeighborsRegressor()

		# Train the model using the training sets
		regr.fit(X_train, y_train)

		# Make predictions using the testing set
		y_pred = regr.predict(X_test)

		# The mean squared error

		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y_test, y_pred))
		r2score.append(r2_score(y_test, y_pred))

	# The mean squared error
	print('Mean squared error Knn: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination Knn: %.2f'
			  % statistics.mean(r2score))

def extratree(X, Y):
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# Create Decision Tree regression object
		   # Fit regression model
		regr= ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
									   random_state=0)

		# Train the model using the training sets
		regr.fit(X_train, y_train)

		# Make predictions using the testing set
		y_pred = regr.predict(X_test)

		# The mean squared error

		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y_test, y_pred))
		r2score.append(r2_score(y_test, y_pred))

	# The mean squared error
	print('Mean squared error ExtraTreesRegressor: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination ExtraTreesRegressor: %.2f'
			  % statistics.mean(r2score))





def DFR(X, Y):
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# Create Decision Tree regression object
		   # Fit regression model
		regr= DecisionTreeRegressor(max_depth=4)

		# Train the model using the training sets
		regr.fit(X_train, y_train)

		# Make predictions using the testing set
		y_pred = regr.predict(X_test)

		# The mean squared error

		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y_test, y_pred))
		r2score.append(r2_score(y_test, y_pred))

	# The mean squared error
	print('Mean squared error DF: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination DF: %.2f'
			  % statistics.mean(r2score))





def ADFR(X, Y):
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	rng = np.random.RandomState(1)
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		# Create Decision Tree regression object
		   # Fit regression model
		regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
						  n_estimators=300, random_state=rng)

		# Train the model using the training sets
		regr.fit(X_train, y_train)

		# Make predictions using the testing set
		y_pred = regr.predict(X_test)

		# The mean squared error
		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y_test, y_pred))
		#print('Coefficient of determination: %.2f'
		  #    % r2_score(y_test, y_pred))
		r2score.append(r2_score(y_test, y_pred))
			# The mean squared error
	print('Mean squared error ADF: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination ADF: %.2f'
			  % statistics.mean(r2score))

def Xgboost_regression(X, y, param):
	datamatrix  = xgb.DMatrix(data=X, label=y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 123)
	xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
				max_depth = 5, alpha = 10, n_estimators = 10)
	params = {"objective": 'reg:squarederror', "eval_metric": 'rmse', 'colsample_bytree' : 0.3, "learning_rate" : 0.1 , "max_depth" : 5, "alpha" : 10, "n_estimators" :10}
	xg_reg.fit(X_train, y_train)
	preds = xg_reg.predict(X_test)
	rmse = np.sqrt(mean_squared_error(y_test, preds))
	r2score = r2_score(y_test, preds)
	print("RMSE: %f" % (rmse))
	print("r2score: %f" % (r2score))
	cv_results = xgb.cv(dtrain=datamatrix, params=params, nfold=5,
					num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
	print(cv_results.head())
	print((cv_results["test-rmse-mean"]).tail(1))
	xg_reg = xgb.train(params=params, dtrain=datamatrix, num_boost_round=10)
	xgb.plot_tree(xg_reg, num_trees=0)
	plt.rcParams['figure.figsize'] = [70, 70]
	plt.rcParams.update({'font.size': 6})
	plt.savefig('tree_regression' + param + '.png')
	plt.close()
	#plt.show()
	plt.rcParams.update({'font.size': 65})
	xgb.plot_importance(xg_reg, max_num_features=10)
	plt.rcParams['figure.figsize'] = [5, 5]
	plt.savefig('xgboost_regression' + param)
	plt.close()

def multioutputregression_methods(X, Y):
	# Fit estimators
	ESTIMATORS = {
		"Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
									   random_state=0),
		"K-nn": KNeighborsRegressor(),
		"Linear regression": LinearRegression(),
		"Ridge": RidgeCV(),
	}
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	rng = np.random.RandomState(1)
	meansquared_error_es = dict()
	r2score_es = dict ()
	for name, estimator in ESTIMATORS.items():
		meansquared_error = []
		r2score = []
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			estimator.fit(X_train, y_train)
			y_pred = estimator.predict(X_test)
			meansquared_error.append(mean_squared_error(y_test, y_pred))
			r2score.append(r2_score(y_test, y_pred))
		meansquared_error_es[name] = statistics.mean(meansquared_error)
		r2score_es[name] = statistics.mean(r2score)
	print(meansquared_error_es)
	print(r2score_es)


def chainregressor(X, Y):
	# Fit estimators
	ESTIMATORS = {
		"Extra trees + chain": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
									   random_state=0),
		"K-nn + chain": KNeighborsRegressor(),
		"Linear regression + chain": LinearRegression(),
		"Ridge + chain": RidgeCV(),
	}
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	rng = np.random.RandomState(1)
	meansquared_error_es = dict()
	r2score_es = dict ()
	for name, estimator in ESTIMATORS.items():
		meansquared_error = []
		r2score = []
		estimator = RegressorChain(estimator)
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			estimator.fit(X_train, y_train)
			y_pred = estimator.predict(X_test)
			meansquared_error.append(mean_squared_error(y_test, y_pred))
			r2score.append(r2_score(y_test, y_pred))
		meansquared_error_es[name] = statistics.mean(meansquared_error)
		r2score_es[name] = statistics.mean(r2score)
	print(meansquared_error_es)
	print(r2score_es)   


def multioutputregression_wrapper(X, Y):
	# Fit estimators
	ESTIMATORS = {
		"Extra trees + wrapper": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
									   random_state=0),
		"K-nn +  wrapper": KNeighborsRegressor(),
		"Linear regression + wrapper ": LinearRegression(),
		"Ridge + wrapper": RidgeCV(),
	}
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	rng = np.random.RandomState(1)
	meansquared_error_es = dict()
	r2score_es = dict ()
	for name, estimator in ESTIMATORS.items():
		meansquared_error = []
		r2score = []
		estimator = MultiOutputRegressor(estimator)
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			estimator.fit(X_train, y_train)
			y_pred = estimator.predict(X_test)
			meansquared_error.append(mean_squared_error(y_test, y_pred))
			r2score.append(r2_score(y_test, y_pred))
		meansquared_error_es[name] = statistics.mean(meansquared_error)
		r2score_es[name] = statistics.mean(r2score)
	print(meansquared_error_es)
	print(r2score_es)

def neauralnetwork_regression(X, y):
	nr.regression(X, y) 



def multioutputregression(df, columns, response_columns):
	multioutputregression_wrapper(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy()))
	chainregressor(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy()))
	multioutputregression_methods(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy()))
	print(mnr.regression(df[columns].to_numpy() ,np.multiply(0.01,df[response_columns].to_numpy())))


def prediction(df, columns, response_columns):
	# predict each target independently
	for response in response_columns:
		print('mean of the data:', statistics.mean(np.multiply(0.01,df[response].to_numpy()))) #percentage
		print('standard deviation of the data:', statistics.stdev(np.multiply(0.01,df[response].to_numpy()))) # percentage
		print(Xgboost_regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy()), response))
		print(Linear_Regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
		print(DFR(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
		print(ADFR(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
		print(Knnn(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
		print(extratree(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
		print(svreg(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
		#print(neauralnetwork_regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))
		print(nrp.regression(df[columns].to_numpy() ,np.multiply(0.01,df[response].to_numpy())))

	multioutputregression(df, columns, response_columns)


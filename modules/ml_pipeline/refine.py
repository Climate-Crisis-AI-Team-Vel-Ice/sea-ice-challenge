
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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, NMF
import multilabelClassification as mc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_regression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from multilabelClassificationMetrics import metrics_precision_recall
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
import uuid
import matplotlib as mpl
from imputer import impute
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
import csv
from functools import reduce



sns.set_style("white")
sns.set_palette("Set2")
plt.style.use('seaborn-white') #sets the size of the charts
#style.use('ggplot')



def adjusted_rsquare(r2,n,p):
	return 1-(1-r2)*(n-1)/(n-p-1)


def plot_features(df, columns, layer):
	missing = []
	for i in columns:
		print('number of missing data ' + i + ' :', (len(df[i]) - df[i].count()))
		print(df[i].min())
		print(df[i].max())
		print(df[i].mean())
		missing.append(len(df[i]) - df[i].count())
		fig = plt.figure(i)
		X = (df[df[i].notna()][i].sort_values().to_numpy()).reshape(-1,1)
		# scaler = MinMaxScaler()
		# scaler.fit((df[df[i].notna()][i].sort_values().to_numpy()).reshape(-1,1))
		# X = scaler.transform((df[df[i].notna()][i].sort_values().to_numpy()).reshape(-1,1))
		#plt.plot(X.reshape(-1,1))
		#ax = df[df[i].notna()][i].sort_values().plot.kde(bw_method=0.3)
		#plt.show()
		#print(len(df[df[i].notna()][i].sort_values().tolist())  
		ax = sns.distplot(X, hist = True, kde = True,
				 kde_kws = {'linewidth': 3},
				 label = i)
		ax2 = ax.twinx()
		sns.boxplot(x=X, ax=ax2)
		ax2.set(ylim=(-.5, 10))

		plt.ylabel('Density')
		plt.title(i)
		plt.savefig('0_' + i +' '+  layer +'.png' )
		plt.close()
	dff = pd.DataFrame(dict(features=columns, number_of_missing=missing))
	sns.barplot(x='features', y = 'number_of_missing', data=dff)
	plt.xticks(
	rotation=45, 
	horizontalalignment='right',
	fontweight='light') 
	plt.title('#missing data ' + layer)
	plt.savefig('missing')
	plt.close()


def reform_targets(response_columns, df):
	for index, value in df['TEXTURE'].items():
		#print(value.split(' '))
		#print(index)
		df['TEXTURE'].update(pd.Series([value.split(' ')[0]], index=[index]))
	print(df['TEXTURE'].unique())
	print(df['TEXTURE'].value_counts())


def plot_y(y,df):
	plt.close()
	df['TEXTURE'].value_counts().plot.bar()
	plt.tight_layout()
	plt.savefig('plot_y')
	plt.close()


def feature_in_onefigure(data, target, df):
	fig,axes =plt.subplots(9,3, figsize=(12, 9)) # 3 columns each containing 9 figures, total 27 features
	ARGILE=df.loc[df['TEXTURE']=='ARGILE', data].to_numpy()
	SABLE=df[df.columns.intersection(data)].loc[df['TEXTURE']=='SABLE', data].to_numpy()
	LOAM=df[df.columns.intersection(data)].loc[df['TEXTURE']=='LOAM', data].to_numpy()

	ax=axes.ravel() # flat axes with numpy ravel
	for i in range(len(data)):
	  _,bins=np.histogram(df[df.columns.intersection(data)].to_numpy()[:,i],bins=40)
	  ax[i].hist(ARGILE[:,i],bins=bins,color='r',alpha=.5)# red color for malignant class
	  ax[i].hist(SABLE[:,i],bins=bins,color='g',alpha=0.3)# alpha is for transparency in the overlapped region 
	  ax[i].hist(LOAM[:,i],bins=bins,color='b',alpha=0.1)
	  ax[i].set_title(df.columns.intersection(data)[i],fontsize=9)
	  ax[i].axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
	  ax[i].set_yticks(())
	ax[0].legend(['ARGILE','SABLE', 'LOAM'],loc='best',fontsize=8)
	plt.tight_layout() # let's make good plots
	plt.savefig('feature')
	plt.close()


def feature(data, target, df):
	#fig,axes =plt.subplots(9,3, figsize=(12, 9)) # 3 columns each containing 9 figures, total 27 features
	ARGILE=df.loc[df['TEXTURE']=='ARGILE', data].to_numpy()
	SABLE=df[df.columns.intersection(data)].loc[df['TEXTURE']=='SABLE', data].to_numpy()
	LOAM=df[df.columns.intersection(data)].loc[df['TEXTURE']=='LOAM', data].to_numpy()

	#ax=axes.ravel() # flat axes with numpy ravel
	for i in range(len(data)):
		fig,ax =plt.subplots(1,1, figsize=(24, 14))
		_,bins=np.histogram(df[df.columns.intersection(data)].to_numpy()[:,i],bins=40)
		ax.hist(ARGILE[:,i],bins=bins,color='r',alpha=.5)# red color for malignant class
		ax.hist(SABLE[:,i],bins=bins,color='g',alpha=0.3)# alpha is for transparency in the overlapped region 
		ax.hist(LOAM[:,i],bins=bins,color='b',alpha=0.1)
		ax.set_title(df.columns.intersection(data)[i],fontsize=25)
		#ax.axes.get_xaxis().set_visible(False) # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
		ax.set_yticks(())
		ax.legend(['ARGILE','SABLE', 'LOAM'],loc='best',fontsize=22)
		plt.tight_layout() # let's make good plots
		plt.savefig('feature' + str(i) + df.columns.intersection(data)[i] )
		plt.close()


def visualization2d(X, y):
	X = preprocessing.scale(X)
	data_subset = X
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(data_subset)
	sns.set_palette("Set2")
	plt.figure(figsize=(16,10))
	sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y
						 ,cmap='Set2', legend='brief') # hue = y
	plt.legend(title='Tsne', loc='upper left', labels=['Low', 'Medium', 'High'])
	plt.title('Tsne Visualization in 2D')
	plt.tight_layout()
	plt.savefig('Tsne')
	plt.close()

def visualization3d(X, y):
	X = preprocessing.scale(X)
	data_subset = X
	tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(data_subset)
	ax = plt.axes(projection='3d')
	for  color, i, target_name in zip(mpl.cm.Set2.colors[:4], [1, 2, 3], ['Low', 'Medium', 'High']):
		ax.scatter(tsne_results[np.where(y.to_numpy() == i), 0], tsne_results[np.where(y.to_numpy() == i), 1], tsne_results[np.where(y.to_numpy() == i), 2], 
			 label=target_name, color=color)
	plt.title('tsne visualization' + " of chemical dataset")
	plt.legend(loc="best", shadow=False, scatterpoints=1)
	plt.tight_layout()
	plt.savefig('3d_tsne')
	plt.close()







#visualization(df[columns].astype('float'), df['TEXTURE'])
def pca(X, dim):
	pca = PCA(dim)
	pca.fit(X)
	print('number of the components', pca.n_components_ )
	print('explained variance', pca.explained_variance_)
	return(pca.fit_transform(X))

def clustering_dbscan(X, labels_true):
	X = StandardScaler().fit_transform(X)
	# Compute DBSCAN
	db = DBSCAN(eps=0.3, min_samples=10).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_
	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	print("Adjusted Rand Index: %0.3f"
		  % metrics.adjusted_rand_score(labels_true, labels))
	print("Adjusted Mutual Information: %0.3f"
		  % metrics.adjusted_mutual_info_score(labels_true, labels))
	print("Silhouette Coefficient: %0.3f"
		  % metrics.silhouette_score(X, labels))
	# Plot result
	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = [plt.cm.Spectral(each)
			  for each in np.linspace(0, 1, len(unique_labels))]
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = [0, 0, 0, 1]

		class_member_mask = (labels == k)

		xy = X[class_member_mask & core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', #, markerfacecolor=tuple(col)
				  markeredgecolor='k' ,markersize=8. ) #markeredgecolor='k',

		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', #, markerfacecolor=tuple(col),
				 markeredgecolor='k', markersize=2)

	plt.title('Estimated number of clusters: %d' % n_clusters_)
	plt.xlabel('First Component')
	plt.ylabel('Second Component')
	plt.legend()
	plt.savefig('clusteringdbscan')
	plt.close()


def clustering_kmeans(X, labels_true):
	#X = df[columns].astype('float')
	#print((X.values.shape)) 
	#print(score)
	X = StandardScaler().fit_transform(X)
	# #############################################################################
	# Compute DBSCAN
	kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
	#core_samples_mask = np.zeros_like(kmeans.labels_, dtype=bool)
	#core_samples_mask[db.core_sample_indices_] = True
	labels = kmeans.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_noise_ = list(labels).count(-1)

	print('Estimated number of clusters: %d' % n_clusters_)
	print('Estimated number of noise points: %d' % n_noise_)
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
	print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
	print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
	print("Adjusted Rand Index: %0.3f"
		  % metrics.adjusted_rand_score(labels_true, labels))
	print("Adjusted Mutual Information: %0.3f"
		  % metrics.adjusted_mutual_info_score(labels_true, labels))
	print("Silhouette Coefficient: %0.3f"
		  % metrics.silhouette_score(X, labels))
	# Plot result
	# Black removed and is used for noise instead.
	#X = pca(X)
	#unique_labels = set(labels)
	sns.set_palette('Set2')
	sns.scatterplot(x=X[:, 0], y=X[:, 1],
						hue=labels_true, style=labels, legend='brief')
	plt.savefig('clustering_kmeans')
	plt.close()



def feature_selection(X, y, data, number_features):
	bestfeatures = SelectKBest(score_func=chi2, k=number_features)
	fit = bestfeatures.fit(X,y)
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X.columns)

	#concat two dataframes for better visualization 
	featureScores = pd.concat([dfcolumns,dfscores],axis=1)
	featureScores.columns = ['X','Score']  #naming the dataframe columns
	print(featureScores.nlargest(number_features,'Score'))  #print 10 best features
	model = ExtraTreesClassifier()
	model.fit(X,y)
	print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

	#plot graph of feature importances for better visualization
	feat_importances = pd.Series(model.feature_importances_, index=X.columns)
	feat_importances.nlargest(number_features).plot(kind='barh')
	plt.rcParams.update({'font.size': 8})
	plt.title('Feature selection ExtraTreesClassifier')
	plt.savefig('feature_selection')
	plt.close()

	#get correlations of each features in dataset
	corrmat = data.corr()
	top_corr_features = corrmat.index
	plt.figure(figsize=(25,25))
	#plot heat map

	plt.rcParams.update({'font.size': 20})
	g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn", )
	plt.savefig('heatmap')
	plt.close()



def incremental_pca(X, y, data):
# Authors: Kyle Kastner
# License: BSD 3 clause
	n_components = 2
	ipca = IncrementalPCA(n_components=n_components, batch_size=10)
	X_ipca = ipca.fit_transform(X)

	pca = PCA(n_components=n_components)
	X_pca = pca.fit_transform(X)
	for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
		#plt.figure()
		for color, i, target_name in zip(mpl.cm.Set2.colors[:4] , [1, 2, 3], ['Low', 'Medium', 'High']):
			plt.scatter(X_transformed[np.where(y.to_numpy() == i), 0], X_transformed[np.where(y.to_numpy() == i), 1],
						color=color, label=target_name)

		if "Incremental" in title:
			err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
			plt.title(title + " of chemical dataset\nMean absolute unsigned error "
					  "%.6f" % err)
		else:
			plt.title(title + " of chemical dataset")
		plt.legend(loc="best", shadow=False, scatterpoints=1)

		plt.savefig('2d_pca' + title)
		plt.close()

	n_components = 3
	ipca = IncrementalPCA(n_components=n_components, batch_size=10)
	X_ipca = ipca.fit_transform(X)

	pca = PCA(n_components=n_components)
	X_pca = pca.fit_transform(X)

	ax = plt.axes(projection='3d')
	for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
		#plt.figure()
		for color, i, target_name in zip(mpl.cm.Set2.colors[:4], [1, 2, 3], ['Low', 'Medium', 'High']):
			ax.scatter(X_transformed[np.where(y.to_numpy() == i), 0], X_transformed[np.where(y.to_numpy() == i), 1], X_transformed[np.where(y.to_numpy() == i), 2],
						color=color, label=target_name)
		if "Incremental" in title:
			err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
			plt.title(title + " of chemical dataset\nMean absolute unsigned error "
					 "%.6f" % err)
		else:
			plt.title(title + " of chemical dataset")
		plt.legend(loc="best", shadow=False, scatterpoints=1)

		plt.savefig('3d_pca' + title)
		plt.close()




def logistic_regression(X, y):
	print(mc.averageprec(X, y))



def xgboost(X, y):
	#X = preprocessing.scale(X)
	datamatrix  = xgb.DMatrix(data=X, label=y)
	n_classes = y.shape[1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 123)
	xg_reg = OneVsRestClassifier(xgb.XGBClassifier())
	params = {"objective": 'multi:softprob', "eval_metric": 'merror','colsample_bytree' : 0.3, "learning_rate" : 0.1 , "max_depth" : 5, "alpha" : 10, "n_estimators" :10, "num_class" :y.shape[1]}
	#xg_reg = xgb.train(dtrain = Xtrain, dtest = X_test, ytrain= y_train, ytest = y_test, params=params)
	#res = xgb.cv(params, X, num_boost_round=1000, nfold=10, seed=seed, stratified=False, early_stopping_rounds=25, verbose_eval=10, show_stdv=True)
	xg_reg.fit(X_train, y_train)
	preds = xg_reg.predict(X_test)
	y_score = xg_reg.predict_proba(X_test)
	#rmse = np.sqrt(mean_squared_error(y_test, preds))
	accuracy = accuracy_score(y_test, preds)
	print("accuracy: %f", (accuracy))
	metrics_precision_recall(y_test, preds, X_test, n_classes, 'XGboost')
	#print("RMSE: %f" % (rmse))
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


def onehotencoder(X):
	ohe = OneHotEncoder() 
	X_ohe = ohe.fit_transform(X) # It returns an numpy array
	return X_ohe.toarray()


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



def feature_selection_regression(X, y, data, number_features, param):
	bestfeatures = SelectKBest(score_func=f_regression, k=number_features)
	fit = bestfeatures.fit(X,y)
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X.columns)

	#concat two dataframes for better visualization 
	featureScores = pd.concat([dfcolumns,dfscores],axis=1)
	featureScores.columns = ['Chemicals','Score']  #naming the dataframe columns
	featureScores = featureScores.sort_values('Score', ascending=False)

	plt.rcParams.update({'font.size': 8})
	featureScores.plot.bar(x='Chemicals', y='Score', rot = 90)
	plt.title(param + ' Feature selection f-classify')
	plt.savefig(param + 'feature_selection_f_classify')
	plt.close()

	bestfeatures = SelectKBest(score_func=mutual_info_regression, k=number_features)
	fit = bestfeatures.fit(X,y)
	dfscores = pd.DataFrame(fit.scores_)
	dfcolumns = pd.DataFrame(X.columns)

	#concat two dataframes for better visualization 
	featureScores = pd.concat([dfcolumns,dfscores],axis=1)
	featureScores.columns = ['Chemicals','Score']  #naming the dataframe columns
	featureScores = featureScores.sort_values('Score', ascending=False)
	featureScores.plot.bar(x='Chemicals', y='Score', rot = 90)
	plt.rcParams.update({'font.size': 8})
	plt.title(param + ' Feature selection mutual_info_regresssion')
	plt.savefig(param + 'feature_selection_mutual_info_regression')
	plt.close()

	model = ExtraTreesRegressor()
	model.fit(X,y)
	# print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
	#plot graph of feature importances for better visualization
	plt.rcParams.update({'font.size': 8})
	feat_importances = pd.Series(model.feature_importances_, index=X.columns)
	feat_importances.nlargest(number_features).plot(kind='barh')
	
	plt.title(param + ' Feature selection ExtraTreesClassifier')
	plt.savefig(param + 'feature_selection_extratrees_regression')
	plt.close()


def correlation_matrix(data):
	#get correlations of each features in dataset
	corrmat = data.corr()
	top_corr_features = corrmat.index
	plt.figure(figsize=(30,30))
	#plot heat map

	plt.rcParams.update({'font.size': 18})
	g=sns.heatmap(data[top_corr_features].corr(), annot=True, cmap="RdYlGn")
	plt.savefig('heatmap_regression', bbox_inches='tight')
	plt.close()


def dimensionality_reduction_f_regression(X, y, param):
		# Split dataset to select feature and evaluate the classifier
	X_train, X_test, y_train, y_test = train_test_split(
			X, y, random_state=0
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
	selector = SelectKBest(f_regression, k=6)
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
	clf = make_pipeline(MinMaxScaler(), SVR(kernel = 'linear'))
	clf.fit(X_train, y_train)
	print('Coefficient of determination without selecting features: {:.3f}'
		  .format(adjusted_rsquare(clf.score(X_test, y_test), X_test.shape[0],X_test.shape[1] )))

	svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
	svm_weights /= svm_weights.sum()

	plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight',
			 edgecolor='black')

	clf_selected = make_pipeline(
			SelectKBest(f_classif, k=6), MinMaxScaler(), SVR(kernel = 'linear')
	)
	clf_selected.fit(X_train, y_train)
	print('Coefficient of determination after univariate feature selection: {:.3f}'
		  .format(adjusted_rsquare(clf_selected.score(X_test, y_test), X_test.shape[0],X_test.shape[1] )))

	svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
	svm_weights_selected /= svm_weights_selected.sum()

	plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
			width=.2, label='SVM weights after selection',
			edgecolor='black')

	plt.title(param + " Comparing feature selection")
	plt.xlabel('Feature number')
	#plt.ylabel('Adjusted R-square')
	plt.yticks(())
	#plt.axis('tight')
	#plt.legend(loc='upper right')
	
	plt.legend(loc='upper left', prop={'size': 13})
	plt.savefig(param + 'dim_reduction_f_regression', bbox_inches='tight')
	plt.close()


def rfeecv(X, y, param):
	# Create the RFE object and compute a cross-validated score.
	svc = LinearRegression()

	rfecv = RFECV(estimator=LinearRegression())
	rfecv.fit(X, y)
	print(rfecv.ranking_)

	print("Optimal number of features : %d" % rfecv.n_features_)
	print(rfecv.ranking_)
	# Plot number of features VS. cross-validation scores
	mpl.rcParams.update(mpl.rcParamsDefault)
	plt.figure()
	plt.xlabel("Number of features selected")
	plt.ylabel("Cross validation score")
	plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
	plt.savefig(param + 'RFECV_regression')

def regplot(x, y, loss, r2score, name, target):
	mpl.rcParams.update(mpl.rcParamsDefault)
	x.sort()
	y.sort()
	fig, ax = plt.subplots()
	ax.scatter(x, y, s=25, zorder=10)
	lims = [
	 np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
	 np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
	]
	# now plot both limits against eachother
	ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
	ax.set_aspect('equal')
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	ax.set_xlabel('true labels')
	ax.set_ylabel('predicted labels')
	ax.set_title(name + ' on  ' + target +  ', loss = ' + "{:.2f}".format(loss) + ', R2 = ' + "{:.2f}".format(r2score))
	filename = target + name + str(uuid.uuid4())
	fig.savefig(filename, bbox_inches='tight')
	fig.clf()
	plt.close()

def linear_Regression(X, Y, target):
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
		# The coefficients
		#print('Coefficients: \n', regr.coef_)
		coefficients = coefficients + regr.coef_
		# The mean squared error
		#print('Mean squared error: %.2f'
		 #     % mean_squared_error(y_test, y_pred))
		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y_test, y_pred))
		#print('Coefficient of determination: %.2f'
		  #    % r2_score(y_test, y_pred))
		r2score.append(r2_score(y_test, y_pred))
	regplot(y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), 'LR', target )



	#print('Coefficients: \n', np.multiply(coefficients,0.2))
	# The mean squared error
	print('Mean squared error: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination: %.2f'
			  % statistics.mean(r2score))
	return regr.predict(X), statistics.mean(meansquared_error), statistics.mean(r2score)



def svreg(X, Y, target):
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
		regplot(y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), 'SVR', target)

	# The mean squared error
	print('Mean squared error SVR: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination SVR: %.2f'
			  % statistics.mean(r2score))

def knnn(X, Y, target):
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
		regplot(y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), 'K-nn', target)

	# The mean squared error
	print('Mean squared error Knn: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination Knn: %.2f'
			  % statistics.mean(r2score))


def plot_importances(X, forest, importances, param, columns):
	std = np.std([tree.feature_importances_ for tree in forest.estimators_],
			 axis=0)
	indices = np.argsort(importances)[::-1]
	#print(indices.tolist())

	indices = list(map(int, indices.tolist()))
	
	#squarer = lambda t: int(t)
	#vfunc = np.vectorize(squarer)
	#indices = vfunc(indices)

	
	ft = [columns[i] for i in indices]
	print(ft)
	# Print the feature ranking
	print("Feature ranking:")

	for f in range(X.shape[1]):
		print("%d. feature %s (%f)" % (f + 1, ft[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	plt.rcParams.update({'font.size': 15})
	plt.figure()
	plt.title(param + " Feature importances")
	plt.bar(range(X.shape[1]), importances[indices],
		   yerr=std[indices], align="center")
	plt.xticks(range(X.shape[1]), ft, rotation = 90)
	plt.xlim([-1, X.shape[1]])
	plt.savefig(param + 'Extratreefeature_importance', bbox_inches='tight')
	plt.close()

def extratree(X, Y, target, columns):
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
	regplot(y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), 'extra tree', target)

	# The mean squared error
	importances = regr.feature_importances_
	plot_importances(X, regr, importances, target, columns)
	print('Mean squared error ExtraTreesRegressor: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination ExtraTreesRegressor: %.2f'
			  % statistics.mean(r2score))
	return regr.predict(X), statistics.mean(meansquared_error), statistics.mean(r2score)



def extratree_forlimon(X, Y, Y2, Y3, target, columns):
	kf = KFold(n_splits=5, shuffle=True) # Define the split - into
	kf_split = kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
	accuracy = []
	r2score = []
	meansquared_error = []
	coefficients = 0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		y2_train, y2_test = Y2[train_index], Y2[test_index]
		y3_test = Y3[test_index]
		# Create Decision Tree regression object
			# Fit regression model

		regr= ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
										random_state=0)

		# Train the model using the training sets
		regr.fit(X_train, y_train)

		# Make predictions using the testing set
		y1_pred = regr.predict(X_test)


		# train model for the second variable
		regr.fit(X_train, y2_train)

		# Make predictions using the testing set
		y2_pred = regr.predict(X_test)

		y_pred = np.multiply(100, np.ones(shape=y1_pred.shape)) - (y1_pred + y2_pred)
		# The mean squared error
		# The coefficient of determination: 1 is perfect prediction
		meansquared_error.append(mean_squared_error(y3_test, y_pred))
		r2score.append(r2_score(y3_test, y_pred))
		regplot(y3_test, y_pred, mean_squared_error(y3_test, y_pred), r2_score(y3_test, y_pred), 'extra tree', target)
	

	print('Mean squared error ExtraTreesRegressor: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination ExtraTreesRegressor: %.2f'
			  % statistics.mean(r2score))


def dFR(X, Y, target):
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
		regplot(y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), 'Random Forest', target)

	# The mean squared error
	print('Mean squared error DF: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination DF: %.2f'
			  % statistics.mean(r2score))





def aDFR(X, Y, target):
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
		regplot(y_test, y_pred, mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred), 'AdaBoost RF ', target)
			# The mean squared error
	print('Mean squared error ADF: %.2f'
			  % statistics.mean(meansquared_error))
	# The coefficient of determination: 1 is perfect prediction
	print('Coefficient of determination ADF: %.2f'
			  % statistics.mean(r2score))

def xgboost_regression(X, y, param, columns):
	datamatrix  = xgb.DMatrix(data=X, label=y, feature_names=columns)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,  random_state = 123)
	xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
				max_depth = 10, alpha = 10, n_estimators = 10)
	params = {"objective": 'reg:squarederror', "eval_metric": 'rmse', 'colsample_bytree' : 0.3, "learning_rate" : 0.1 , "max_depth" : 5, "alpha" : 10, "n_estimators" :10}
	xg_reg.fit(X_train, y_train)
	preds = xg_reg.predict(X_test)

	preds = preds[~np.isnan(y_test)]
	y_test = y_test[~np.isnan(y_test)]

	mse = (mean_squared_error(y_test, preds))
	r2score = r2_score(y_test, preds)
	#regplot(y_test, preds, mean_squared_error(y_test, preds), r2_score(y_test, preds), 'XGBoost', param)
	print("MSE XGBoost: %f" % (mse))
	print("r2score XGBoost: %f" % (r2score))
	#cv_results = xgb.cv(dtrain=datamatrix, params=params, nfold=5,
	#               num_boost_round=50,early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
	#print(cv_results.head())
	#print((cv_results["test-rmse-mean"]).tail(1))
	xg_reg = xgb.train(params=params, dtrain=datamatrix, num_boost_round=10)
	#xgb.train(params=params, dtrain=datamatrix, num_boost_round=10)
	xgb.plot_tree(xg_reg, num_trees=0)
	#plt.rcParams['figure.figsize'] = [50, 50]
	plt.rcParams.update({'font.size': 100})
	plt.savefig('tree_regression' + param + '.eps', bbox_inches='tight', format='eps')
	plt.close()
	#plt.show() 
	plt.rcParams.update({'font.size': 65})
	xgb.plot_importance(xg_reg, max_num_features=5, importance_type='weight')
	plt.rcParams['figure.figsize'] = [5, 5]
	plt.savefig('xgboost_regression' + param)
	plt.close()



def multioutputregression(X, Y):
	# Fit estimators
	ESTIMATORS = {
		"Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
										random_state=0),
		"K-nn": KNeighborsRegressor(),
		"Linear regression": LinearRegression(),
		"Ridge": RidgeCV(),
	}
	kf = KFold(n_splits=2, shuffle=True) # Define the split - into
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

			# meansquared_error.append(mean_squared_error(y_test, y_pred))
			# r2score.append(r2_score(y_test, y_pred))

			print(name , mean_squared_error(y_test, y_pred, multioutput='raw_values'))
			print(name, r2_score(y_test, y_pred, multioutput='raw_values'))


	# 	meansquared_error_es[name] = statistics.mean(meansquared_error)
	# 	r2score_es[name] = statistics.mean(r2score)
	# print(meansquared_error_es)
	# print(r2score_es)


def chainregressor(X, Y):
	# Fit estimators
	ESTIMATORS = {
		"Extra trees + chain": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
										random_state=0),
		"K-nn + chain": KNeighborsRegressor(),
		"Linear regression + chain": LinearRegression(),
		"Ridge + chain": RidgeCV(),
	}
	kf = KFold(n_splits=2, shuffle=True) # Define the split - into
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
		estimator = RegressorChain(estimator, order = [0, 1, 2])
		for train_index, test_index in kf.split(X):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = Y[train_index], Y[test_index]
			estimator.fit(X_train, y_train)
			y_pred = estimator.predict(X_test)
			
			print(name , mean_squared_error(y_test, y_pred, multioutput='raw_values'))
			print(name, r2_score(y_test, y_pred, multioutput='raw_values'))

			# meansquared_error.append(mean_squared_error(y_test, y_pred))
			# r2score.append(r2_score(y_test, y_pred))

	# 	meansquared_error_es[name] = statistics.mean(meansquared_error)
	# 	r2score_es[name] = statistics.mean(r2score)
	# print(meansquared_error_es)
	# print(r2score_es)   



def multioutputregression_wrapper(X, Y):
	# Fit estimators
	ESTIMATORS = {
		"Extra trees + wrapper": ExtraTreesRegressor(n_estimators=10, max_features=X.shape[1],
										random_state=0),
		# "K-nn +  wrapper": KNeighborsRegressor(),
		# "Linear regression + wrapper ": LinearRegression(),
		# "Ridge + wrapper": RidgeCV(),
	}
	kf = KFold(n_splits=2, shuffle=True) # Define the split - into
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

			print(name , mean_squared_error(y_test, y_pred, multioutput='raw_values'))
			print(name, r2_score(y_test, y_pred, multioutput='raw_values'))

		meansquared_error_es[name] = statistics.mean(meansquared_error)
		r2score_es[name] = statistics.mean(r2score)
	importances = estimator.feature_importances_
	plot_importances(X, estimator, importances, 'multi', columns)
	print(meansquared_error_es)
	print(r2score_es)
	return estimator.predict(X), meansquared_error_es['Extra trees + wrapper'], r2score_es['Extra trees + wrapper']



		
def neauralnetwork_classification(X, y):
	nn_nn.classification(X, y)   


def neauralnetwork_regression(X, y, target):
	nr.regression(X, y, target)  



def classification_texture():
	columns = []
	response_columns= []

	#SABLE
	#LIMON
	#ARGILE

	#response_columns = ['SABLETG', 'SABLEG',   'SABLEM',   'SABLEF',   'SABLETF',  'SABLE' ,'LIMONF',  'LIMONM',   'LIMONG',   'LIMON' ,'ARGILE',  'BATTANCE'  ,'CONDHYD', 'ML',   'MN']
	
	response_columns = ['TEXTURE']
	columns = ['MO',    'N' ,'INDICE20' ,'PHEAU', 'PHSMP',  'KECH', 'CAECH' , 'MGECH',  'NAECH' , 'HECH', 'CEC', 'PM3', 'MNM3', 'CUM3'  ,'FEM3' ,'ALM3' ,'BM3', 'ZNM3', 'PBM3', 'MOM3', 'CDM3', 'COM3', 'CRM3'  ,'KM3'  ,'CAM3' ,'MGM3',    'NAM3']
	columns_ECH = columns[:11]
	columns_M3 =  columns[11:]


	columns = columns_M3

	# component = ['B', 'Co', 'Mn', 'Cu', 'P', 'K', 'Fe', 'Al', 'Cr', 'Na', 'Ca', 'Mo', 'Mg']
	# chemicals_M3 = [columns[0]] + columns[11:]
	# chemicals_ECH = [columns[0]] + columns[5:10]

	try:
		columns.remove('N')
		columns.remove('INDICE20')
		columns.remove('PHSMP')
	except:
		pass


	for columns in [columns]:


		df = pd.read_excel('Tabi_ML.xlsx', usecols = columns + response_columns)
		
		pd.set_option('display.max_columns', 500)
		pd.set_option('display.width', 1000)
		print(df.describe())
		print(len(columns))
		print(df['TEXTURE'].unique())
		print(df.shape)
		print(df.head(3))
		#plot_features(df, columns)
		print(df['TEXTURE'].value_counts(normalize=True) * 100)
		df = df.dropna(inplace=True)
		reform_targets(response_columns, df)
		print(df['TEXTURE'].value_counts(normalize=True) * 100)
		#response_columns = ['SABLETG', 'SABLEG',   'SABLEM',   'SABLEF',   'SABLETF',  'SABLE' ,'LIMONF',  'LIMONM',   'LIMONG',   'LIMON' ,'ARGILE',  'BATTANCE'  ,'CONDHYD', 'ML',   'MN']
		df = df[df['TEXTURE'].notna()]
		plot_y(response_columns, df)
		print(df.shape)
		print(df.head(3))
		scaler = MinMaxScaler()
		feature(columns, response_columns, df)



		#clustering_dbscan(pca(df[columns].to_numpy()), df['TEXTURE'])      
		#clustering_kmeans(pca(df[columns].to_numpy()), df['TEXTURE'])


		#pca(df[columns])
		#visualization2d(df[columns], df['TEXTURE'])


		response_column = df['TEXTURE']
		label_encoder = LabelEncoder()
		label_encoder = label_encoder.fit(response_column)
		label_encoded_y = label_encoder.transform(response_column)
		df.drop('TEXTURE', axis = 1, inplace = True)
		df['TEXTURE'] = label_encoded_y


		print(df['TEXTURE'].unique())
		print(df['TEXTURE'].value_counts())


		#visualization3d(df[columns], df['TEXTURE'])
		#incremental_pca(df[columns], df['TEXTURE'], df)
		#feature_selection(df[columns], df['TEXTURE'], df, 5)



		a = np.array(df['TEXTURE'].to_numpy().reshape(1, df['TEXTURE'].to_numpy().shape[0]))
		b = np.zeros((a.size, a.max()+1))
		b[np.arange(a.size),a] = 1


		neauralnetwork_classification(df[columns].to_numpy() ,b)
		logistic_regression(df[columns] , b)
		svmclassifier(df[columns].to_numpy() ,df['TEXTURE'].to_numpy())
		xgboost(df[columns] , b)


		dimensionality_reduction_f_classify(df[columns].to_numpy() , df['TEXTURE'].to_numpy())
		compare_methods(df[columns].to_numpy() , df['TEXTURE'].to_numpy())

# import geopandas as gpd
# import geoplot as gplt
# import geoplot.crs as gcrs

def geopandas_(df):

	qc = gpd.read_file('BDPPAD_v03_2015_s_20161207.shp')
	ax = qc.plot(color='white', edgecolor='black')
	# gplt.polyplot(qc, projection=gcrs.AlbersEqualArea())
	qc.plot(color='white', edgecolor='black')
	plt.show()
	print(qc.head())

	# gplt.polyplot(contiguous_usa)
	# gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea())



	gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.xcoord, y=df.ycoord))
	# gdf.dropna(inplace=True)
	print(gdf.head())

	# ax = gplt.polyplot(qc[qc.AN == 2015], projection=gcrs.AlbersEqualArea())
	# gplt.pointplot(gdf, ax=ax)

	gdf.plot(ax=ax)
	plt.savefig('test')
	# gplt.choropleth(gdf, shade=True, cmap='Reds',projection=gplt.crs.AlbersEqualArea(), hue='MVA')
	# geoplot.polyplot(boroughs, ax=ax, zorder=1, hue='MVA')
	# ax = gplt.kdeplot(gdf,projection=gcrs.AlbersEqualArea(), cmap='Reds', hue='MVA', clip=qc.geometry)
	# gplt.pointplot(df, hue='MVA', legend=True, ax=ax)
	# plt.show()

def embedding_layerbylayer(df):
	df1 = df[GROUPE.x ==1]
	df2 = df[GROUPE.x == 2]
	df3 = df[GROUPE.x == 3]






import pcatsne as viz

def prediction():
	# results={'llwithout':{'linearregression':{}, 'NN':{}, 'Extratrees':{}}, 'llwith'={}}
	columns = []
	response_columns= []

	#SABLE
	#LIMON
	#ARGILE

	# # response_columns = ['SABLE', 'LIMON', 'ARGILE']
	# # response_columns = ['MVA', 'CONDHYD', 'DMP', 'POROTOT', 'MO']
	# response_columns = ['MVA', 'POROTOT1', 'POROTOT3', 'PORODRAI1', 'PORODRAI3', 'CH_cm_h'] #DMP
	# # response_columns=['MVA']
	# columns = ['PCMO', 'PM3', 'CEC', 'MNM3', 'CUM3'  ,'FEM3' ,'ALM3' ,'BM3'  ,'KM3'  ,'CAM3' ,'MGM3', 'ARGILE', 'SABLE', 'LIMON', 'CentreEp', 'PHSMP', 'PHEAU']
	# # columns = []
	
	# df1 = pd.read_csv('Couche_Inv1990tot.csv', usecols = ['IDEN2.x'] + ['IDEN3'] + ['GROUPE.x'] + ['Couche'] + columns + response_columns, encoding='latin-1')
	# df2 = pd.read_csv('Site_Inv1990.csv', usecols = ['IDEN2'] + ['xcoord', 'ycoord'], encoding='latin-1')
	# df3 = pd.read_csv('Champ_Inv1990.csv', usecols = ['IDEN3', 'Culture_1']  , encoding='latin-1')
	# df5 = pd.read_csv('Inv90_RawClim.csv' , encoding='latin-1' , delimiter = ';' ,  usecols = ['IDchamps', 'TMOY', 'PCP', 'ETP', 'TMAX', 'TMIN'],  thousands=',')
	# print(df5.columns)
	# df5[['TMOY', 'PCP', 'ETP', 'TMAX', 'TMIN']] = df5[['TMOY', 'PCP', 'ETP', 'TMAX', 'TMIN']].apply(pd.to_numeric)
	# grouped_df = df5.groupby('IDchamps')
	# print(grouped_df)
	# df6 = grouped_df['TMOY', 'PCP', 'ETP', 'TMAX', 'TMIN'].mean().reset_index()
	# print(df6)
	# df7 = df3.merge(df6, left_on='IDEN3', right_on='IDchamps', left_index = True)
	# del df7['IDchamps']
	# # print(df1.columns)
	# # print(df2.columns)
	# print(df7.columns)

	

	# # df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
	# df4 = df1.merge(df2, left_on='IDEN2.x', right_on='IDEN2').reindex(columns=['IDEN2', 'IDEN3', 'GROUPE.x', 'Couche' ] + ['xcoord', 'ycoord'] + columns + response_columns)
	# df = df4.merge(df7, left_on='IDEN3', right_on='IDEN3')
	# # geopandas_(df)
	
	# # print('stop')
	# print(df.columns)
	# # plot_features(df, columns, 'Alltogether')


	# print(df['Culture_1'].unique())
	# print(df['Culture_1'].value_counts())

	# # df.rename(columns={'PCMO': 'OM', 'PM3': 'P', 'MNM3': 'MN', 'CUM3': 'CU', 'FEM3': 'FE', 'ALM3': 'AL', 'BM3': 'B', 'KM3': 'K', 'CA3': 'CA', 'MGM3': 'MG'}, inplace=True)
	
	# # use pd.concat to join the new columns with your original dataframe
	# df = pd.concat([df,pd.get_dummies(df['Culture_1'])],axis=1)

	# # now drop the original 'country' column (you don't need it anymore)
	# df.drop(['Culture_1'],axis=1, inplace=True)
	# # print(df.columns)

	# columns = columns + ['xcoord', 'ycoord'] + ['1-prairie',
	#    '2-céréales', '3-maïs-grain', '4-pommes de terre', '5-maïs-ensilage',
	#    '6-autres', 'TMOY', 'PCP', 'ETP'] #, 'TMAX', 'TMIN'


	# for column in columns:
	# 	df[column].fillna((df[column].mean()), inplace=True)
	
	# df = df[df['GROUPE.x'] == 1]


	# cc = columns.copy()
	# dfcouche1 = df[df['Couche'] == 1]
	# dfcouche2 = df[df['Couche'] == 2]
	# dfcouche3 = df[df['Couche'] == 3]

	# print(dfcouche1.describe(include='all').transpose())
	# print(dfcouche2.describe(include='all').transpose())
	# print(dfcouche3.describe(include='all').transpose())

	# for column in columns + response_columns:
	# 	print(column, dfcouche1[column].skew())
	# 	print(column, dfcouche2[column].skew())
	# 	print(column, dfcouche3[column].skew())
	df1 = pd.read_csv('hashed1_Relevant_GEN.csv')
	df2 = pd.read_csv('profit_MuMu.csv')

	df2.fillna(0, inplace=True)
	df2 = df2.assign(Total=(df2['60'] +  df2['59'])*0.5)
	print(df2.head())
	print(df1.head())
	columns = list(df1.columns)
	columns.remove('ANM_ID')
	df2 = df2[['ANM_ID','Total']]
	df = df2.merge(df1,left_on='ANM_ID', right_on='ANM_ID')
	
	df.to_csv('gen_valacta_profit_cum.csv',  header=True)
	response_columns_predictin = ['Total']
	response_columns = ['Total']
	df.drop(['ANM_ID'], axis=1, inplace=True)
	df.dropna(inplace = True)
	df = df.sample(n=10000, random_state=1)
	scaler = MinMaxScaler()
	df[columns] = scaler.fit_transform(df[columns])
	total = pd.qcut( df['Total'], 3, labels=[ 1, 2,3])
	total = pd.DataFrame(total, columns = ['Total'])

	df.drop(['Total'],axis=1, inplace=True)

	df = pd.concat([df,total],axis=1)
	
	df.sample(n=10000, random_state=1)
	print(df.head())

	# viz.main(df[columns], df['Total'], ['low','medium','high'], "profit")
	# viz.main(df[columns], df['GROUPE.x'], [1,2,3], "group")


	feature_selection(df[columns], df['Total'], df, 10)
	pca = PCA(n_components=10)
	pca_df = pca.fit_transform(df[columns])

	columns = ['MEDIAN_SUSP', 'UDDER_TEXTURE', 'EBV_FAT', 'ANGULARITY', 'FORE_ATTACH', 'RVALUE', 'HEEL_DEPTH', 'REAR_ATTACH_WIDTH', 'EBV_PROT', 'REAR_TEAT_PLACE']

	df[columns] = pd.DataFrame(pca_df)
	df.dropna(inplace = True)

	# print(df3.head())

	# print(pca.components_)
	# print(pca.components_)

	# columns  = df.columns[pca.components_]

	visualization3d(df[columns], df['Total'])
	incremental_pca(df[columns], df['Total'], df)

	plot_features(df, columns, 'profit')
	correlation_matrix(df[columns])



	a = np.array(df['Total'].to_numpy().reshape(1, df['Total'].to_numpy().shape[0]))
	b = np.zeros((a.size, a.max()+1))
	b[np.arange(a.size),a] = 1


	neauralnetwork_classification(df[columns].to_numpy() ,b)
	logistic_regression(df[columns] , b)
	# svmclassifier(df[columns].to_numpy() ,df['Total'].to_numpy())
	xgboost(df[columns] , b)


	dimensionality_reduction_f_classify(df[columns].to_numpy() , df['Total'].to_numpy())
	compare_methods(df[columns].to_numpy() , df['Total'].to_numpy())



	for responsee in response_columns:

		# dfff = [0, 0 , 0]
		# dfff[0] =  dfcouche1
		# dfff[1] =  dfcouche2[['IDEN2', responsee]]
		# dfff[2] =  dfcouche3[['IDEN2', responsee]]





		# plot_features(dfcouche1, columns, 'couche1')
		# plot_features(dfcouche2, columns, 'couche2')
		plot_features(df, columns, 'couche3')



		# print(dfcouche1['IDEN2'].count())
		# print(dfcouche1['IDEN2'].nunique())

		# print(dfcouche2['IDEN2'].count())
		# print(dfcouche2['IDEN2'].nunique())

		# print(dfcouche3['IDEN2'].count())
		# print(dfcouche3['IDEN2'].nunique())


		# df_ = dfcouche1.merge(dfcouche2[['IDEN2', 'MVA']], 'inner', on=['IDEN2'], suffixes=['_1', '_2'])
		# df_ = dfcouche1.merge(dfcouche2['IDEN2', 'MVA'], 'inner', on=['IDEN2'], suffixes=['_2'])

		# MVA = [df_['MVA_1'], df_['MVA_2']]

		# df_ = reduce(lambda  left,right: pd.merge(left,right,on=['IDEN2'],how='inner'), dfff)
		# df_ = dfcouche1.merge(dfcouche2[['IDEN2', 'MVA']], 'inner', on=['IDEN2'], suffixes=['_1', '_2'])
		# df_ = df_.merge(dfcouche3['IDEN2', 'MVA'], 'inner', on=['IDEN2'], suffixes=['_3'])
		# df_.rename ({responsee + '_x': responsee + '_1', responsee + '_y': responsee + '_2', responsee : responsee + '_3'}, axis=1, inplace=True)

		# print(df_.columns)
		# MVA = [df_['MVA_1'], df_['MVA_2'], df_['MVA_3']]

		# print(df_['MVA_1'].count())
		# print(df_['MVA_2'].count())

		# print(df_['IDEN2'].nunique())


		# np.save('X.npy', df[columns].to_numpy())
		# np.save('y.npy', df['GROUPE.x'].to_numpy())

		
		# raise Exception("stopppp!")
		correlation_matrix(df)

		# viz.main(df[columns], df['Couche'], [1,2,3], "couche")
		# viz.main(df[columns], df['GROUPE.x'], [1,2,3], "group")

		
		
		# pd.set_option('display.max_columns', 500)
		# pd.set_option('display.width', 1000)
		# print(df[response_columns].describe())
		# print(df.shape)
		# plot_features(df, response_columns)
		# df.dropna(inplace = True)
		# print(df.describe())
		

		# mva = ['N', 'PHSMP', 'INDICE20', 'ALM3', 'KM3', 'CAM3', 'MOM3', 'MGM3', 'FEM3', 'CDM3', 'COM3', 'PM3', 'CUM3', 'BM3', 'CRM3', 'NAM3', 'MNM3', 'ZNM3', 'PBM3']
		# condhyd= ['N', 'MOM3', 'MGM3', 'CAM3', 'BM3', 'KM3', 'INDICE20', 'PHSMP', 'ALM3', 'PM3', 'CUM3', 'FEM3', 'CRM3', 'NAM3', 'CDM3', 'ZNM3', 'PBM3', 'COM3', 'MNM3']
		# dmp = ['PM3', 'N', 'MGM3', 'INDICE20', 'KM3', 'CAM3', 'BM3', 'ALM3', 'MOM3', 'CUM3', 'FEM3', 'PBM3', 'NAM3', 'PHSMP', 'CRM3', 'MNM3', 'ZNM3', 'CDM3', 'COM3']
		# portot = ['PHSMP', 'N', 'ALM3', 'INDICE20', 'MOM3', 'KM3', 'CAM3', 'MGM3', 'CUM3', 'FEM3', 'COM3', 'CRM3', 'NAM3', 'PM3', 'BM3', 'CDM3', 'MNM3', 'ZNM3', 'PBM3']
		# mo = ['N', 'INDICE20', 'PHSMP', 'KM3', 'MGM3', 'MOM3', 'COM3', 'CDM3', 'CUM3', 'FEM3', 'CRM3', 'ALM3', 'PM3', 'ZNM3', 'CAM3', 'BM3', 'MNM3', 'PBM3', 'NAM3']
		
		# selected_columns={'MVA': mva[0:6], 'CONDHYD': condhyd[0:6], 'DMP': dmp[0:6] , 'POROTOT': portot[0:6], 'MO': mo[0:6] }
		
		# try:
		# 	columns.remove('N')
		# 	columns.remove('INDICE20')
		# 	columns.remove('PHSMP')
		# except:
		# 		pass
		
		results = []
		summaryofdata = []
		results.append(['layer', 'target', 'MSE', 'Rsquare'])
		summaryofdata.append(['layer', 'target', 'mean', 'stdev', 'number'])

		response_columns_0 = [responsee + '_1', responsee + '_2', responsee + '_3']


		# for response in response_columns_0:
			# dff = dfff.copy()
			# columns = cc
			# for i in range(3):
			# 	df1 = dff[i].copy()
			# 	#columns = selected_columns[response]
			# 	# break
			# 	# df = pd.read_excel('Tabi_ML.xlsx', usecols = columns + [response])
			# 	df = df_.copy()
			# 	# print(df.isnull().sum())

		df1 = pd.read_csv('hashed1_Relevant_GEN.csv')
		df2 = pd.read_csv('profit_MuMu.csv')
		df2.fillna(0, inplace=True)
		df2 = df2.assign(Total=lambda d: d.sum(1))
		print(df2.head())
		print(df1.head())
		columns = list(df1.columns)
		columns.remove('ANM_ID')
		df2 = df2[['ANM_ID','Total']]
		df = df2.merge(df1,left_on='ANM_ID', right_on='ANM_ID')
		print(df.head())

		df.to_csv('gen_valacta_profit_cum.csv',  header=True)
		response_columns = ['Total']
		response = 'Total'
		df.drop(['ANM_ID'], axis=1, inplace=True)
		df.dropna(inplace = True)
		# df.dropna(inplace = True, subset=[response])

		# print(response + 'mean of the data:', statistics.mean(df[response].to_numpy())) #percentage
		# print(response + 'standard deviation of the data:', statistics.stdev(df[response].to_numpy())) # percentage
		# print('shape:', df.shape)
		# summaryofdata.append(['layer' + str(i), response, statistics.mean(df[response].to_numpy()), statistics.stdev(df[response].to_numpy()), str(df.shape[0])])
		# print(i)
		# print(df.columns)
		df.dropna(inplace=True)
		X = df[columns].to_numpy()
		print(X.shape)

		#do pca first
		dim = 6
		X = PCA(dim).fit(X).transform(X)

		scaler = MinMaxScaler().fit(X)
		X = scaler.transform(X)

		Y = df[response].to_numpy()
		# Y = df[response_columns].to_numpy()
		# print(Y.shape)



		'''scaler =  MinMaxScaler().fit(Y)
		Y = scaler.transform(Y)'''
		# impute(X, Y, response)

		for column in columns:
			df[column].fillna((df[column].mean()), inplace=True)


		# X = df[columns].to_numpy()
		mpl.rcParams.update(mpl.rcParamsDefault)
		# feature_selection_regression (df[columns], df[response], df, 10, response)
		# dimensionality_reduction_f_regression(X, Y, response)
		# rfeecv(X, Y, response)
		# xgboost_regression(X, Y , response, columns)


		pr, error, rsquare  = linear_Regression(X , Y, response)
		print(response, error, rsquare)
		results.append([ response, str(error) , str(rsquare), 'Linear Regression'])


		# dFR(X , Y, response)
		# aDFR(X ,Y, response)
		# knnn(X ,Y, response)


		prediction, error, rsquare  = extratree(X ,Y, response, columns)
		print(response, error, rsquare)
		results.append([ response, str(error) , str(rsquare), 'ExtraTrees'])


		svreg(X ,Y, response)
		neauralnetwork_regression(X , Y, response)


		# prediction, error, rsquare = nrp.regression(X , Y, response)
		# print(response, error, rsquare)
		# results.append([ response, str(error) , str(rsquare), 'Neural Netwrok'])

		# print(prediction.shape)
		# pred = pd.DataFrame(prediction.flatten() , columns=[response + str(i)] )
		# pred['IDEN2'] = df['IDEN2'].to_numpy()

		# response_columns = ['MVA_1', 'MVA_2']
		# df = df_.copy()
		# df.dropna(inplace = True, subset=response_columns_0)
		pr, error, rsquare  = multioutputregression_wrapper(df[columns].to_numpy() ,df[response_columns].to_numpy())
		# results.append(['layer', response_columns, str(error) , str(rsquare), 'ExtraTrees'])


		# results.append(['layer' + str(i), response, , 'Rsquare']).


		prediction, error, rsquare  = mnr.regression(df[columns].to_numpy() ,df[response_columns].to_numpy())

		# results.append(['layer', response_columns, str(error) , str(rsquare), 'Neural Netwrok'])


		chainregressor(df[columns].to_numpy() ,df[response_columns].to_numpy())
		
		# print(prediction.shape)
		# pred = pd.DataFrame(prediction , columns=[response + str(i) for response in response_columns] )
		# pred['IDEN2'] = df['IDEN2'].to_numpy()

		# if i+1 < 3:
		# 	df_ = df.merge( pred, 'outer').reindex()
		# 	dff[i+1] = df_.copy()
		# 	print(dff[i+1].columns)
		# 	# columns = columns + [response + str(i) for response in response_columns]
		# 	columns = columns + [response + str(i)]


	with open('results.csv', 'w') as myfile:
		wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(results)


	# with open('summaryofdata.csv', 'w') as myfile:
	# 	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	# 	wr.writerow(summaryofdata)


		# df = pd.read_excel('Tabi_ML.xlsx', usecols = columns + response_columns)
		# for column in columns:
		# 	df[column].fillna((df[column].mean()), inplace=True)

		# df.dropna(inplace=True)
		# print(df.isnull().sum())
		# # extratree_forlimon(df[columns].to_numpy(), df['ARGILE'].to_numpy(), df['SABLE'].to_numpy(), df['LIMON'].to_numpy(), 'Limon', columns )
		# df.dropna(inplace=True)



		# multioutputregression_wrapper(df[columns].to_numpy() ,df[response_columns].to_numpy())

		# chainregressor(df[columns].to_numpy() ,df[response_columns].to_numpy())

		# multioutputregression(df[columns].to_numpy() ,df[response_columns].to_numpy())

		# print(mnr.regression(df[columns].to_numpy() ,df[response_columns].to_numpy()))





#classification_texture()
prediction()

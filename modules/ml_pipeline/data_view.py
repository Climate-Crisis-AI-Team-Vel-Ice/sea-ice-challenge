#visualization_and_clustering.py
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA, NMF
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib as mpl


sns.set_style("white")
sns.set_palette("Set2")
plt.style.use('seaborn-white')


def visualization2d(X, y):
	X = preprocessing.scale(X)
	data_subset = X
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(data_subset)
	sns.set_palette("Set2")
	plt.figure(figsize=(16,10))
	sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=y
						 ,cmap='Set2', legend='brief') # hue = y
	plt.legend(title='Tsne', loc='upper left', labels=['ARGILE', 'LIMON', 'LOAM', 'SABLE'])
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
	for  color, i, target_name in zip(mpl.cm.Set2.colors[:4], ['ARGILE', 'LIMON', 'LOAM', 'SABLE'], ['ARGILE', 'LIMON', 'LOAM', 'SABLE']):
		ax.scatter(tsne_results[np.where(y.to_numpy() == i), 0], tsne_results[np.where(y.to_numpy() == i), 1], tsne_results[np.where(y.to_numpy() == i), 2], 
			 label=target_name, color=color)
	plt.title('tsne visualization' + " of chemical dataset")
	plt.legend(loc="best", shadow=False, scatterpoints=1)
	plt.tight_layout()
	plt.savefig('3d_tsne')
	plt.close()


def pca(X):
	pca = PCA(2)
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

	X = StandardScaler().fit_transform(X)
	# #############################################################################
	# Compute DBSCAN
	kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
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
	featureScores.columns = ['Chemicals','Score']  #naming the dataframe columns
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
		for color, i, target_name in zip(mpl.cm.Set2.colors[:4] , ['ARGILE', 'LIMON', 'LOAM', 'SABLE'], ['ARGILE', 'LIMON', 'LOAM', 'SABLE']):
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
		for color, i, target_name in zip(mpl.cm.Set2.colors[:4], ['ARGILE', 'LIMON', 'LOAM', 'SABLE'], ['ARGILE', 'LIMON', 'LOAM', 'SABLE']):
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


def clustering(df, columns, response_columns):    
	clustering_dbscan(pca(df[columns].to_numpy()), df[response_columns])      
	clustering_kmeans(pca(df[columns].to_numpy()), df[response_columns])
	pca(df[columns])

def encoded_labels(df, response_columns):
	response_column = df[response_columns]
	label_encoder = LabelEncoder()
	label_encoder = label_encoder.fit(response_column)
	label_encoded_y = label_encoder.transform(response_column)

	df.drop(response_columns, axis = 1, inplace = True)
	# Put whatever series you want in its place
	df['TEXTURE'] = label_encoded_y

def visualization(df, columns, response_columns):
	visualization2d(df[columns], df[response_columns])
	visualization3d(df[columns], df[response_columns])
	incremental_pca(df[columns], df[response_columns], df)

def feature_selection_dimensionality_reduction(df, columns, response_columns, num_features_selected):
	feature_selection(df[columns], df[response_columns], df, num_features_selected)


def data_view(df, columns, response_columns):
	df = df.dropna()
	visualization(df, columns, response_columns)
	clustering(df, columns, response_columns)
	encoded_labels(df, response_columns)
	feature_selection_dimensionality_reduction(df, columns, response_columns, 5)
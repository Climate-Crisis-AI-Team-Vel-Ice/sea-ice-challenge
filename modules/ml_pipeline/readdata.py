import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib as mpl

sns.set_style("white")
sns.set_palette("Set2")
plt.style.use('seaborn-white')


def plot_features(df, columns):
	missing = []
	for i in columns:
		print('number of missing data ' + i + ' :', (len(df[i]) - df[i].count()))
		missing.append(len(df[i]) - df[i].count())
		fig = plt.figure(i)

		scaler = MinMaxScaler()
		scaler.fit((df[df[i].notna()][i].sort_values().to_numpy()).reshape(-1,1))
		X = scaler.transform((df[df[i].notna()][i].sort_values().to_numpy()).reshape(-1,1))
		ax = sns.distplot(X, hist = True, kde = True,
				 kde_kws = {'linewidth': 3},
				 label = i)
		ax2 = ax.twinx()
		sns.boxplot(x=X, ax=ax2)
		ax2.set(ylim=(-.5, 10))

		plt.ylabel('Density')
		plt.title(i)
		#plt.savefig('0_' + i + '.png' )
		#plt.close()
	dff = pd.DataFrame(dict(features=columns, number_of_missing=missing))
	sns.barplot(x='features', y = 'number_of_missing', data=dff)
	plt.xticks(
	rotation=45,
	horizontalalignment='right',
	fontweight='light')
	plt.title('#missing data')
	#plt.savefig('missing')
	#plt.close()


def reform_targets(response_columns, df):
	for index, value in df[response_columns].items():
		#print(value.split(' '))
		#print(index)
		df[response_columns].update(pd.Series([value.split(' ')[0]], index=[index]))
	print(df[response_columns].unique())
	print(df[response_columns].value_counts())


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
	ARGILE=df.loc[df['TEXTURE']=='ARGILE', data].to_numpy()
	SABLE=df[df.columns.intersection(data)].loc[df['TEXTURE']=='SABLE', data].to_numpy()
	LOAM=df[df.columns.intersection(data)].loc[df['TEXTURE']=='LOAM', data].to_numpy()

	for i in range(len(data)):
		fig,ax =plt.subplots(1,1, figsize=(24, 14))
		_,bins=np.histogram(df[df.columns.intersection(data)].to_numpy()[:,i],bins=40)
		ax.hist(ARGILE[:,i],bins=bins,color='r',alpha=.5)# red color for malignant class
		ax.hist(SABLE[:,i],bins=bins,color='g',alpha=0.3)# alpha is for transparency in the overlapped region
		ax.hist(LOAM[:,i],bins=bins,color='b',alpha=0.1)
		ax.set_title(df.columns.intersection(data)[i],fontsize=25)
		ax.set_yticks(())
		ax.legend(['ARGILE','SABLE', 'LOAM'],loc='best',fontsize=22)
		plt.tight_layout() # let's make good plots
		plt.savefig('feature' + str(i) + df.columns.intersection(data)[i] )
		plt.close()


def summary_of_data(df, columns, response_columns):

	#pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)

	#print(df.describe())
	print('number of features:', len(columns))
	print('number of targets:', len(response_columns))
	print('labels:', response_columns)
	print('number of rows:', df.shape[0])

	"""
	plot_features(df, columns)
	print('labels distribution:')
	print(df[response_columns].value_counts(normalize=True) * 100)

	df = df.dropna()


	#aggregate the targets from 13 rto 4 groups
	reform_targets(response_columns, df)
	print('labels distribution aggregated:')

	print(df[response_columns].value_counts(normalize=True) * 100)

	plot_y(response_columns, df)
	feature(columns, response_columns, df)
	"""
	return df


def read_data(filename, columns, response_columns):
	if isinstance(response_columns, list):
		df = pd.read_excel(filename, usecols = columns + response_columns)
		return df
	else:
		df = pd.read_excel(filename, usecols = columns + [response_columns])
		df = summary_of_data(df, columns, response_columns)
		return df

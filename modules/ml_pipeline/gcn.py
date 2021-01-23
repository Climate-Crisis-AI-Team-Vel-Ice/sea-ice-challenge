#gcn.py

import torch
import networkx as nx 
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import pandas as pd
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset



def dataloader():
	G = nx.read_gpickle("spatial_graph.gpickle") 
	data = from_networkx(G)
	# print('here')
	# print(data.num_nodes)
	# print(data.num_edges)
	# print(data.num_node_features)
	# print(data)
	# print(data['x'])
	# print(data['y'])
	return data, G

def add_features_(data, G):
	# response_columns = ['MVA', 'POROTOT1', 'POROTOT3', 'PORODRAI1', 'PORODRAI3', 'CH_cm_h', 'DMP' ] #DMP
	response_columns=['MVA']
	columns = ['PCMO', 'PM3', 'CEC', 'MNM3', 'CUM3'  ,'FEM3' ,'ALM3' ,'BM3'  ,'KM3'  ,'CAM3' ,'MGM3', 'ARGILE', 'SABLE', 'LIMON', 'CentreEp', 'PHSMP', 'PHEAU']
	# columns = []
	
	df1 = pd.read_csv('Couche_Inv1990tot.csv', usecols = ['IDEN2.x'] + ['IDEN3'] + ['GROUPE.x'] + ['Couche'] + columns + response_columns, encoding='latin-1')
	df2 = pd.read_csv('Site_Inv1990.csv', usecols = ['IDEN2'] + ['xcoord', 'ycoord'], encoding='latin-1')
	df3 = pd.read_csv('Champ_Inv1990.csv', usecols = ['IDEN3', 'Culture_1']  , encoding='latin-1')
	# print(df1.columns)
	# print(df2.columns)
	# print(df3.columns)
	

	# df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
	df4 = df1.merge(df2, left_on='IDEN2.x', right_on='IDEN2').reindex(columns=['IDEN2', 'IDEN3', 'GROUPE.x', 'Couche' ] + ['xcoord', 'ycoord'] + columns + response_columns)
	df = df4.merge(df3, left_on='IDEN3', right_on='IDEN3')
	# geopandas_(df)
	
	# print('stop')
	# print(df.columns)
	# plot_features(df, columns, 'Alltogether')




	# df.rename(columns={'PCMO': 'OM', 'PM3': 'P', 'MNM3': 'MN', 'CUM3': 'CU', 'FEM3': 'FE', 'ALM3': 'AL', 'BM3': 'B', 'KM3': 'K', 'CA3': 'CA', 'MGM3': 'MG'}, inplace=True)
	
	# use pd.concat to join the new columns with your original dataframe
	df = pd.concat([df,pd.get_dummies(df['Culture_1'])],axis=1)

	# now drop the original 'country' column (you don't need it anymore)
	df.drop(['Culture_1'],axis=1, inplace=True)
	# print(df.columns)

	columns = columns + ['xcoord', 'ycoord'] + ['1-prairie',
	   '2-céréales', '3-maïs-grain', '4-pommes de terre', '5-maïs-ensilage',
	   '6-autres']



	for column in columns:
		df[column].fillna((df[column].mean()), inplace=True)

	for column in response_columns:
		df[column].fillna((df[column].mean()), inplace=True)
		df[column] = (df[column] - df[column].mean())/ df[column].std()
	

	dfcouche1 = df[df['Couche'] == 1]

	df = dfcouche1
	x_features = torch.tensor(df[columns].to_numpy()[list(G.nodes()), :], dtype = torch.float)
	y_targets = torch.tensor(df[response_columns].to_numpy()[list(G.nodes()), :], dtype = torch.float)

	data['x'] = x_features
	data['y'] = y_targets
	# print(data['x'])
	# print(data['y'])
	return data

data, G = dataloader()
data = add_features_(data, G)

    
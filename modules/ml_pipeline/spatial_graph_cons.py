from scipy import spatial
import pandas as pd
# import geopandas as gpd
import numpy as np
import networkx as nx
import pickle


response_columns = ['MVA', 'POROTOT1', 'POROTOT3', 'PORODRAI1', 'PORODRAI3', 'CH_cm_h', 'DMP' ] #DMP
# response_columns=['MVA']
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


print(df['Culture_1'].unique())
print(df['Culture_1'].value_counts())

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


dfcouche1 = df[df['Couche'] == 1]
df = dfcouche1

# id_xy = pd.read_csv('Site_Inv1990.csv', usecols = ['IDEN2'], dtype=str)['IDEN2'].to_numpy()
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.xcoord, y=df.ycoord)).to_crs('+proj=robin')
# df[['xcoord', 'ycoord']] = df[['xcoord', 'ycoord']].convert_objects(convert_numeric=True)
dfx = df['xcoord'].to_numpy()
dfy = df['ycoord'].to_numpy()
# id_xy = df['ycoord'].to_numpy()
# centroids = df.to_numpy()
centroids = np.column_stack([dfx, dfy])
print(dfx.shape)
print(dfy.shape)
# We define the range
radius=600
# Like in the previous example we populate the KD-tree
kdtree = spatial.cKDTree(centroids)
neigh_list = {}
edge_list =[]

# We cycle on every point and calculate its neighbours 
# with the function query_ball_point
spatial_graph = nx.Graph()

for m, g in enumerate(centroids):
	neigh_list[m] = (kdtree.query_ball_point(g, r=radius))
	for item in neigh_list[m][:-1]:
		spatial_graph.add_edge(m, item)
nx.write_gpickle(spatial_graph, "spatial_graph.gpickle")

print(spatial_graph.number_of_nodes())
print(spatial_graph.number_of_edges())

# Dump graph
with open("spatial_graph.pickle", 'wb') as f:
    pickle.dump(spatial_graph, f)

# f.close()


# print(neigh_list)
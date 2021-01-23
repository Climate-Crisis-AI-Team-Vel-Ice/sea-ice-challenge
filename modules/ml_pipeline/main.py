#main.py    
	
from data_view import data_view
from soil_classification import classification
from soil_prediction import prediction
from readdata import read_data
import sys

def main():
	response_columns= [] 
	columns = []    
	filename = sys.argv[1]
	#columns = sys.argv[2].strip('[]').split(',')
	response_columns_classification = sys.argv[2].strip('[]').split(', ')
	response_columns_predictin = sys.argv[3].strip('[]').split(', ')

	#features
	columns = ['MO',    'N' ,'INDICE20' ,'PHEAU', 'PHSMP',  'KECH', 'CAECH' , 'MGECH',  'NAECH' , 'HECH', 'CEC', 'PM3', 'MNM3', 'CUM3'  ,'FEM3' ,'ALM3' ,'BM3', 'ZNM3', 'PBM3', 'MOM3', 'CDM3', 'COM3', 'CRM3'  ,'KM3'  ,'CAM3' ,'MGM3',    'NAM3']
	columns_M3 = columns[1:5] + [columns[9]] + columns[11:] 
	columns = columns_M3

	#single target classification
	for response_target in response_columns_classification:
		df = read_data(filename, columns, response_target)
		data_view(df, columns, response_target)
		classification(df, columns, response_target)
	#regression
	df = read_data(filename, columns, response_columns_predictin)
	df.dropna(inplace=True)
	prediction(df, columns, response_columns_predictin)


if __name__ == "__main__":
	main()





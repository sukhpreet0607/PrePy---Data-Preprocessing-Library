from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
import pickle
import time
import os
import argparse
import pandas as pd 



def encode_data(data, path): 
	''' Encode objects values into numerical values '''
	
	for col in data.columns : 
		if data[col].dtype == 'object' : 

			encoder = LabelEncoder()
			encoder.fit(data[col])
			data[col] = encoder.transform(data[col]) 

			# Save encoder
# 			encoder_file = path + '/' + col +'_encoder.sav'
# 			pickle.dump(encoder, open(encoder_file, 'wb'))
	return data 



def scale(data): 
	''' Scaling 
	Return scaled dataframe''' 

	scaler = StandardScaler()
	scaler.fit(data)
	data = scaler.transform(data)
	# Save the scaler 
# 	scalerfile = 'scaler.sav'
# 	pickle.dump(scaler, open(scalerfile, 'wb'))
	return data



def principal_comp_analysis(data, nb_comp, label_column): 
	''' Principal components analysis transformation 
	Return transformed dataframe with nb_comp features'''

	features_col = []
	for col in data.columns: 
		if col != label_column:
			features_col.append(col)

	features = data.drop([label_column], axis = 1)
	label_data = data[[label_column]]

	pca = PCA(n_components = nb_comp)
	pca.fit(features)
	data_pc = pca.transform(features)

	columns = []
	for i in range(nb_comp): 
		columns.append('pc{}'.format(i+1))

	df = pd.DataFrame(data = data_pc, columns = columns)
	df = pd.concat([df, label_data], axis =1)

	return df 
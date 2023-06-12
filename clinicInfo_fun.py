import pandas as pd


def clinicInfo_idcolumn(clinicInfo_path):
	df = pd.read_csv(clinicInfo_path)
	sorted_df = df.sort_values(by='Px')
	id_column = sorted_df.iloc[:, 0]
	sorted_df.set_index('Px', inplace=True)

	return id_column


def clinicInfo_values(clinicInfo_path, pxID):
	df = pd.read_csv(clinicInfo_path)
	sorted_df = df.sort_values(by='Px')
	sorted_df.set_index('Px', inplace=True)
	#Get Clinical Info
	side_value = sorted_df.loc[pxID, 'Side']
	upper_value = sorted_df.loc[pxID, 'Upper']
	
	return side_value,upper_value
	

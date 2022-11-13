import pandas as pd
import numpy as np
from replace_missing import replace_missing4 

#Loading the data
path_p09gaf13 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p09gaf13_2022-01-25.csv'
path_p09gaf14 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p09gaf14_2022-01-25.csv'
path_p09gaf15 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p09gaf15_2022-01-25.csv'
path_p09gaf16 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p09gaf16_2022-01-25.csv'

df_p09gaf13 = pd.read_csv(path_p09gaf13, sep=';')
df_p09gaf14 = pd.read_csv(path_p09gaf14, sep=';')
df_p09gaf15 = pd.read_csv(path_p09gaf15, sep=';')
df_p09gaf16 = pd.read_csv(path_p09gaf16, sep=';')

# Dropping irrelevant columns 
df_p09gaf13 = df_p09gaf13.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p09gaf13_id','p09gaf13_protocol','p09gaf13_project','p09gaf13_measurement','p09gaf13_notes', 'p09gaf13_location','p09gaf13_invited_at','p09gaf13_emailed_at','p09gaf13_open_from','p09gaf13_non_response','p09gaf13_compl_by', 'p09gaf13_started_at','p09gaf13_completed_at','p09gaf13_variant','p09gaf13_anonymous'})
df_p09gaf14 = df_p09gaf14.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p09gaf14_id','p09gaf14_protocol','p09gaf14_project','p09gaf14_measurement','p09gaf14_notes', 'p09gaf14_location','p09gaf14_invited_at','p09gaf14_emailed_at','p09gaf14_open_from','p09gaf14_non_response','p09gaf14_compl_by', 'p09gaf14_started_at','p09gaf14_completed_at','p09gaf14_variant','p09gaf14_anonymous'})
df_p09gaf15 = df_p09gaf15.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p09gaf15_id','p09gaf15_protocol','p09gaf15_project','p09gaf15_measurement','p09gaf15_notes', 'p09gaf15_location','p09gaf15_invited_at','p09gaf15_emailed_at','p09gaf15_open_from','p09gaf15_non_response','p09gaf15_compl_by', 'p09gaf15_started_at','p09gaf15_completed_at','p09gaf15_variant','p09gaf15_anonymous'})
df_p09gaf16 = df_p09gaf16.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p09gaf16_id','p09gaf16_protocol','p09gaf16_project','p09gaf16_measurement','p09gaf16_notes', 'p09gaf16_location','p09gaf16_invited_at','p09gaf16_emailed_at','p09gaf16_open_from','p09gaf16_non_response','p09gaf16_compl_by', 'p09gaf16_started_at','p09gaf16_completed_at','p09gaf16_variant','p09gaf16_anonymous'})

df_p09gaf13 = df_p09gaf13.rename(columns={'p09gaf13_date': 'Date','p09gaf13_922':'GAF Symptoms','p09gaf13_923':'GAF disability'})
df_p09gaf14 = df_p09gaf14.rename(columns={'p09gaf14_date': 'Date','p09gaf14_922':'GAF Symptoms','p09gaf14_923':'GAF disability'})
df_p09gaf15 = df_p09gaf15.rename(columns={'p09gaf15_date': 'Date','p09gaf15_922':'GAF Symptoms','p09gaf15_923':'GAF disability'})
df_p09gaf16 = df_p09gaf16.rename(columns={'p09gaf16_date': 'Date','p09gaf16_922':'GAF Symptoms','p09gaf16_923':'GAF disability'})

# Sorting data to only keep first measurements available for every patient_id
df_p09gaf13['Date']= pd.to_datetime(df_p09gaf13['Date'], format='%d-%m-%Y %H:%M:%S')
df_p09gaf13 = df_p09gaf13.sort_values(by="Date") 
df_p09gaf13 = df_p09gaf13.drop_duplicates("patient_id", keep='first')
df_p09gaf14['Date']= pd.to_datetime(df_p09gaf14['Date'], format='%d-%m-%Y %H:%M:%S')
df_p09gaf14 = df_p09gaf14.sort_values(by="Date") 
df_p09gaf14 = df_p09gaf14.drop_duplicates("patient_id", keep='first')
df_p09gaf15['Date']= pd.to_datetime(df_p09gaf15['Date'], format='%d-%m-%Y %H:%M:%S')
df_p09gaf15 = df_p09gaf15.sort_values(by="Date") 
df_p09gaf15 = df_p09gaf15.drop_duplicates("patient_id", keep='first')
df_p09gaf16['Date']= pd.to_datetime(df_p09gaf16['Date'], format='%d-%m-%Y %H:%M:%S')
df_p09gaf16 = df_p09gaf16.sort_values(by="Date") 
df_p09gaf16 = df_p09gaf16.drop_duplicates("patient_id", keep='first')

df_p09gaf_merge = df_p09gaf13.merge(df_p09gaf14, on='patient_id', how='outer', suffixes=('13', '14'))
df_p09gaf_merge = df_p09gaf_merge.merge(df_p09gaf15, on='patient_id', how='outer', suffixes=('', '15'))
df_p09gaf_merge = df_p09gaf_merge.merge(df_p09gaf16, on='patient_id', how='outer', suffixes=('15', '16'))
df_p09gaf_merger = df_p09gaf_merge.replace('>', '', regex=True).replace('<', '', regex=True)

# Only keep the first measurement available which is not NaN 
df_p09gaf_merger["GAF_symptoms_all"] = df_p09gaf_merger.apply(lambda x: replace_missing4(x['GAF Symptoms13'], x["GAF Symptoms14"], x['GAF Symptoms15'], x["GAF Symptoms16"]), axis = 1)
df_p09gaf_merger["GAF_disability_all"] = df_p09gaf_merger.apply(lambda x: replace_missing4(x['GAF disability13'], x["GAF disability14"], x['GAF disability15'], x["GAF disability16"]), axis = 1)

#Keep only combined columns
df_p09gaf = df_p09gaf_merger[['patient_id','GAF_symptoms_all','GAF_disability_all']]

#Create a file containing the relevant variables 
df_p09gaf.to_csv('df_p09gaf.csv')
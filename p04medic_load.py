import pandas as pd
import numpy as np
from replace_missing import replace_missing4 

# Loading the data, define the necessary path  
path_p04medic13 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p04medic13_2022-01-25.csv'
path_p04medic14 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p04medic14_2022-01-25.csv'
path_p04medic15 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p04medic15_2022-01-25.csv'
path_p04medic16 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p04medic16_2022-01-25.csv'

df_p04medic13 = pd.read_csv(path_p04medic13, sep=';')
df_p04medic14 = pd.read_csv(path_p04medic14, sep=';')
df_p04medic15 = pd.read_csv(path_p04medic15, sep=';')
df_p04medic16 = pd.read_csv(path_p04medic16, sep=';')

# Dropping irrelevant columns 
df_p04medic13 = df_p04medic13.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p04medic13_id','p04medic13_protocol','p04medic13_project','p04medic13_measurement','p04medic13_notes', 'p04medic13_location','p04medic13_invited_at','p04medic13_emailed_at','p04medic13_open_from','p04medic13_non_response','p04medic13_compl_by', 'p04medic13_started_at','p04medic13_completed_at','p04medic13_variant','p04medic13_anonymous'})
df_p04medic14 = df_p04medic14.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p04medic14_id','p04medic14_protocol','p04medic14_project','p04medic14_measurement','p04medic14_notes', 'p04medic14_location','p04medic14_invited_at','p04medic14_emailed_at','p04medic14_open_from','p04medic14_non_response','p04medic14_compl_by', 'p04medic14_started_at','p04medic14_completed_at','p04medic14_variant','p04medic14_anonymous'})
df_p04medic15 = df_p04medic15.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p04medic15_id','p04medic15_protocol','p04medic15_project','p04medic15_measurement','p04medic15_notes', 'p04medic15_location','p04medic15_invited_at','p04medic15_emailed_at','p04medic15_open_from','p04medic15_non_response','p04medic15_compl_by', 'p04medic15_started_at','p04medic15_completed_at','p04medic15_variant','p04medic15_anonymous'})
df_p04medic16 = df_p04medic16.drop(columns={'gender', 'birth_year', 'roqua_id','hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label','p04medic16_id','p04medic16_protocol','p04medic16_project','p04medic16_measurement','p04medic16_notes', 'p04medic16_location','p04medic16_invited_at','p04medic16_emailed_at','p04medic16_open_from','p04medic16_non_response','p04medic16_compl_by', 'p04medic16_started_at','p04medic16_completed_at','p04medic16_variant','p04medic16_anonymous'})

df_p04medic13 = df_p04medic13.rename(columns={'p04medic13_date': 'Date', 'p04medic13_m1_291':'Clozapine','p04medic13_m1_292':'Olanzapine','p04medic13_m1_293':'Risperidon','p04medic13_m1_294':'Quetiapine','p04medic13_m1_295':'Aripiprazol','p04medic13_m1_296':'Haloperidol','p04medic13_m1_1272':'Other antipsychotics1','p04medic13_m2_1272':'Other antipsychotics2'})
df_p04medic14 = df_p04medic14.rename(columns={'p04medic14_date': 'Date', 'p04medic14_m1_291':'Clozapine','p04medic14_m1_292':'Olanzapine','p04medic14_m1_293':'Risperidon','p04medic14_m1_294':'Quetiapine','p04medic14_m1_295':'Aripiprazol','p04medic14_m1_296':'Haloperidol','p04medic14_m1_1272':'Other antipsychotics1','p04medic14_m2_1272':'Other antipsychotics2'})
df_p04medic15 = df_p04medic15.rename(columns={'p04medic15_date': 'Date', 'p04medic15_m1_291':'Clozapine','p04medic15_m1_292':'Olanzapine','p04medic15_m1_293':'Risperidon','p04medic15_m1_294':'Quetiapine','p04medic15_m1_295':'Aripiprazol','p04medic15_m1_296':'Haloperidol','p04medic15_m1_1272':'Other antipsychotics1','p04medic15_m2_1272':'Other antipsychotics2'})
df_p04medic16 = df_p04medic16.rename(columns={'p04medic16_date': 'Date', 'p04medic16_m1_291':'Clozapine','p04medic16_m1_292':'Olanzapine','p04medic16_m1_293':'Risperidon','p04medic16_m1_294':'Quetiapine','p04medic16_m1_295':'Aripiprazol','p04medic16_m1_296':'Haloperidol','p04medic16_m1_1272':'Other antipsychotics1','p04medic16_m2_1272':'Other antipsychotics2'})

# Sorting data to only keep first measurements available for every patient_id
df_p04medic13['Date'] = pd.to_datetime(df_p04medic13['Date'], format='%d-%m-%Y %H:%M:%S')
df_p04medic13 = df_p04medic13.sort_values(by="Date") 
df_p04medic13 = df_p04medic13.drop_duplicates("patient_id", keep='first')
df_p04medic14['Date'] = pd.to_datetime(df_p04medic14['Date'], format='%d-%m-%Y %H:%M:%S')
df_p04medic14 = df_p04medic14.sort_values(by="Date") 
df_p04medic14 = df_p04medic14.drop_duplicates("patient_id", keep='first')
df_p04medic15['Date'] = pd.to_datetime(df_p04medic15['Date'], format='%d-%m-%Y %H:%M:%S')
df_p04medic15 = df_p04medic15.sort_values(by="Date") 
df_p04medic15 = df_p04medic15.drop_duplicates("patient_id", keep='first')
df_p04medic16['Date'] = pd.to_datetime(df_p04medic16['Date'], format='%d-%m-%Y %H:%M:%S')
df_p04medic16 = df_p04medic16.sort_values(by="Date") 
df_p04medic16 = df_p04medic16.drop_duplicates("patient_id", keep='first')

df_p04medic_merge = df_p04medic13.merge(df_p04medic14, on='patient_id', how='outer', suffixes=('13','14'))
df_p04medic_merge = df_p04medic_merge.merge(df_p04medic15, on='patient_id', how='outer', suffixes=('','15'))
df_p04medic_merge = df_p04medic_merge.merge(df_p04medic16, on='patient_id', how='outer', suffixes=('15','16'))
df_p04medic_merger = df_p04medic_merge.replace('>', '', regex=True).replace('<', '', regex=True)

# Only keep the first measurement available which is not NaN 
df_p04medic_merger["Clozapine_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Clozapine13'], x["Clozapine14"], x['Clozapine15'], x["Clozapine16"]), axis = 1)
df_p04medic_merger["Olanzapine_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Olanzapine13'], x["Olanzapine14"], x['Olanzapine15'], x["Olanzapine16"]), axis = 1)
df_p04medic_merger["Risperidon_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Risperidon13'], x["Risperidon14"], x['Risperidon15'], x["Risperidon16"]), axis = 1)
df_p04medic_merger["Quetiapine_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Quetiapine13'], x["Quetiapine14"], x['Quetiapine15'], x["Quetiapine16"]), axis = 1)
df_p04medic_merger["Aripiprazol_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Aripiprazol13'], x["Aripiprazol14"], x['Aripiprazol15'], x["Aripiprazol16"]), axis = 1)
df_p04medic_merger["Haloperidol_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Haloperidol13'], x["Haloperidol14"], x['Haloperidol15'], x["Haloperidol16"]), axis = 1)
df_p04medic_merger["Other_antipsychotics1_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Other antipsychotics113'], x["Other antipsychotics114"], x['Other antipsychotics115'], x["Other antipsychotics116"]), axis = 1)
df_p04medic_merger["Other_antipsychotics2_all"] = df_p04medic_merger.apply(lambda x: replace_missing4(x['Other antipsychotics213'], x["Other antipsychotics214"], x['Other antipsychotics215'], x["Other antipsychotics216"]), axis = 1)

#Remove individual columns and only keep combined column
df_p04medics = df_p04medic_merger[['patient_id','Clozapine_all','Olanzapine_all','Risperidon_all','Quetiapine_all','Aripiprazol_all','Haloperidol_all',"Other_antipsychotics1_all",'Other_antipsychotics2_all']]
# Combine all individual medication to one variable called 'Antipsychotics'
df_p04medics['Antipsychotics'] = df_p04medics[list(df_p04medics.columns)].sum(axis=1)
# Only keep values 0 and 1, meaning no antipsychotics or antipsychotics
df_p04medics['Antipsychotics'].values[df_p04medics['Antipsychotics'] > 1] = 1
df_p04medic = df_p04medics[['patient_id','Antipsychotics']]

#Create a file containing the relevant variables 
df_p04medic.to_csv('df_p04medic.csv')
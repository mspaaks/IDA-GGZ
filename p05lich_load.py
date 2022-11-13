import pandas as pd
import numpy as np

# Loading the data 
path_p05lich13 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p05lich13_2022-01-25.csv'
path_p05lich14 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p05lich14_2022-01-25.csv'
path_p05lich15 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p05lich15_2022-01-25.csv'
path_p05lich16 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\p05lich16_2022-01-25.csv'

df_p05lich13 = pd.read_csv(path_p05lich13, sep=';')
df_p05lich14 = pd.read_csv(path_p05lich14, sep=';')
df_p05lich15 = pd.read_csv(path_p05lich15, sep=';')
df_p05lich16 = pd.read_csv(path_p05lich16, sep=';')

# Dropping irrelevant columns
df_p05lich13 = df_p05lich13.drop(columns={'p05lich13_650','p05lich13_bui_i','p05lich13_pol_i','p05lich13_sys_i','p05lich13_bmi_i','p05lich13_id','p05lich13_anonymous','p05lich13_variant', 'p05lich13_completed_at','p05lich13_compl_by','p05lich13_started_at','p05lich13_open_from','p05lich13_non_response','p05lich13_protocol', 'p05lich13_emailed_at', 'p05lich13_measurement', 'p05lich13_invited_at', 'p05lich13_notes', 'p05lich13_location', 'gender', 'birth_year', 'roqua_id', 'hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label', 'p05lich13_project', 'p05lich13_sys', 'p05lich13_645', 'p05lich13_646'})
df_p05lich14 = df_p05lich14.drop(columns={'p05lich14_650','p05lich14_bui_i','p05lich14_pol_i','p05lich14_sys_i','p05lich14_bmi_i','p05lich14_id','p05lich14_anonymous','p05lich14_variant', 'p05lich14_completed_at','p05lich14_compl_by','p05lich14_started_at','p05lich14_open_from','p05lich14_non_response','p05lich14_protocol', 'p05lich14_emailed_at', 'p05lich14_measurement', 'p05lich14_invited_at', 'p05lich14_notes', 'p05lich14_location', 'gender', 'birth_year', 'roqua_id', 'hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label', 'p05lich14_project', 'p05lich14_645', 'p05lich14_646','p05lich14_648'})
df_p05lich15 = df_p05lich15.drop(columns={'p05lich15_650','p05lich15_bui_i','p05lich15_pol_i','p05lich15_sys_i','p05lich15_bmi_i','p05lich15_id','p05lich15_anonymous','p05lich15_variant', 'p05lich15_completed_at','p05lich15_compl_by','p05lich15_started_at','p05lich15_open_from','p05lich15_non_response','p05lich15_protocol', 'p05lich15_emailed_at', 'p05lich15_measurement', 'p05lich15_invited_at', 'p05lich15_notes', 'p05lich15_location', 'gender', 'birth_year', 'roqua_id', 'hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label', 'p05lich15_project', 'p05lich15_sys', 'p05lich15_645', 'p05lich15_646','p05lich15_1085a','p05lich15_1085b','p05lich15_800'})
df_p05lich16 = df_p05lich16.drop(columns={'p05lich16_650','p05lich16_bui_i','p05lich16_pol_i','p05lich16_sys_i','p05lich16_bmi_i','p05lich16_id','p05lich16_anonymous','p05lich16_variant', 'p05lich16_completed_at','p05lich16_compl_by','p05lich16_started_at','p05lich16_open_from','p05lich16_non_response','p05lich16_protocol', 'p05lich16_emailed_at', 'p05lich16_measurement', 'p05lich16_invited_at', 'p05lich16_notes', 'p05lich16_location', 'gender', 'birth_year', 'roqua_id', 'hide_pii_from_researchers', 'hide_values_from_professionals', 'respondent_id', 'respondent_type', 'respondent_label', 'p05lich16_project', 'p05lich16_sys', 'p05lich16_645', 'p05lich16_646','p05lich16_1085a','p05lich16_1085b','p05lich16_800','p05lich16_646a','p05lich16_648a'})


# Rename the remaining columns
df_p05lich13 = df_p05lich13.rename(columns={'p05lich13_date': 'Date', 'p05lich13_bmi': 'BMI', 'p05lich13_pol':'Heartrate','p05lich13_bui':'Waist','p05lich13_643':'Height','p05lich13_1085':'Weight','p05lich13_647':'Irregular HR','p05lich13_648':'BP sys','p05lich13_649':'BP dias','p05lich13_1110':'Akathisia','p05lich13_1112':'Acute dystonia','p05lich13_1114':'Tardive dyskinesia','p05lich13_1116':'Parkinsonism'})  
df_p05lich14 = df_p05lich14.rename(columns={'p05lich14_date': 'Date', 'p05lich14_bmi': 'BMI', 'p05lich14_pol':'Heartrate','p05lich14_bui':'Waist','p05lich14_643':'Height','p05lich14_1085':'Weight','p05lich14_647':'Irregular HR','p05lich14_sys':'BP sys','p05lich14_649':'BP dias','p05lich14_1110':'Akathisia','p05lich14_1112':'Acute dystonia','p05lich14_1114':'Tardive dyskinesia','p05lich14_1116':'Parkinsonism'})
df_p05lich15 = df_p05lich15.rename(columns={'p05lich15_date': 'Date', 'p05lich15_bmi': 'BMI', 'p05lich15_pol':'Heartrate','p05lich15_bui':'Waist','p05lich15_643':'Height','p05lich15_1085':'Weight','p05lich15_647':'Irregular HR','p05lich15_648':'BP sys','p05lich15_649':'BP dias','p05lich15_1110':'Akathisia','p05lich15_1112':'Acute dystonia','p05lich15_1114':'Tardive dyskinesia','p05lich15_1116':'Parkinsonism'})
df_p05lich16 = df_p05lich16.rename(columns={'p05lich16_date': 'Date', 'p05lich16_bmi': 'BMI', 'p05lich16_pol':'Heartrate','p05lich16_bui':'Waist','p05lich16_643':'Height','p05lich16_1085':'Weight','p05lich16_647':'Regular HR','p05lich16_648':'BP sys','p05lich16_649':'BP dias','p05lich16_1110':'Akathisia','p05lich16_1112':'Acute dystonia','p05lich16_1114':'Tardive dyskinesia','p05lich16_1116':'Parkinsonism'})  

df_p05lich13 = df_p05lich13.sort_values(by="Date") 
df_p05lich13 = df_p05lich13.drop_duplicates("patient_id", keep='first')
df_p05lich14['Date'] = pd.to_datetime(df_p05lich14['Date'], format='%d-%m-%Y %H:%M:%S')
df_p05lich14 = df_p05lich14.sort_values(by="Date") 
df_p05lich14 = df_p05lich14.drop_duplicates("patient_id", keep='first')
df_p05lich15['Date'] = pd.to_datetime(df_p05lich15['Date'], format='%d-%m-%Y %H:%M:%S')
df_p05lich15 = df_p05lich15.sort_values(by="Date") 
df_p05lich15 = df_p05lich15.drop_duplicates("patient_id", keep='first')
df_p05lich16['Date'] = pd.to_datetime(df_p05lich16['Date'], format='%d-%m-%Y %H:%M:%S')
df_p05lich16 = df_p05lich16.sort_values(by="Date") 
df_p05lich16 = df_p05lich16.drop_duplicates("patient_id", keep='first')

df_p05lich_merge = df_p05lich13.merge(df_p05lich14, on='patient_id', how='outer', suffixes=('13', '14'))
df_p05lich_merge = df_p05lich_merge.merge(df_p05lich15, on='patient_id', how='outer', suffixes=('', '15'))
df_p05lich_merge = df_p05lich_merge.merge(df_p05lich16, on='patient_id', how='outer', suffixes=('15', '16'))

# Fillna in column of left df with right df and drop no longer needed columns, avoid multiple columns for same feature
df_p05lich_merge['BMIa'] = df_p05lich_merge['BMI13'].fillna(df_p05lich_merge['BMI14'])
df_p05lich_merge['BMIb'] = df_p05lich_merge['BMIa'].fillna(df_p05lich_merge['BMI15'])
df_p05lich_merge['BMI'] = df_p05lich_merge['BMIb'].fillna(df_p05lich_merge['BMI16'])
df_p05lich_merge = df_p05lich_merge.drop(['BMIa','BMIb','BMI13','BMI14','BMI15','BMI16'], axis=1)

df_p05lich_merge['Heartratea'] = df_p05lich_merge['Heartrate13'].fillna(df_p05lich_merge['Heartrate14'])
df_p05lich_merge['Heartrateb'] = df_p05lich_merge['Heartratea'].fillna(df_p05lich_merge['Heartrate15'])
df_p05lich_merge['Heartrate'] = df_p05lich_merge['Heartrateb'].fillna(df_p05lich_merge['Heartrate16'])
df_p05lich_merge = df_p05lich_merge.drop(['Heartratea','Heartrateb','Heartrate13','Heartrate14','Heartrate15','Heartrate16'], axis=1)

df_p05lich_merge['Waista'] = df_p05lich_merge['Waist13'].fillna(df_p05lich_merge['Waist14'])
df_p05lich_merge['Waistb'] = df_p05lich_merge['Waista'].fillna(df_p05lich_merge['Waist15'])
df_p05lich_merge['Waist'] = df_p05lich_merge['Waistb'].fillna(df_p05lich_merge['Waist16'])
df_p05lich_merge = df_p05lich_merge.drop(['Waista','Waistb','Waist13','Waist14','Waist15','Waist16'], axis=1)

df_p05lich_merge['Heighta'] = df_p05lich_merge['Height13'].fillna(df_p05lich_merge['Height14'])
df_p05lich_merge['Heightb'] = df_p05lich_merge['Heighta'].fillna(df_p05lich_merge['Height15'])
df_p05lich_merge['Height'] = df_p05lich_merge['Heightb'].fillna(df_p05lich_merge['Height16'])
df_p05lich_merge = df_p05lich_merge.drop(['Heighta','Heightb','Height13','Height14','Height15','Height16'], axis=1)

df_p05lich_merge['Weighta'] = df_p05lich_merge['Weight13'].fillna(df_p05lich_merge['Weight14'])
df_p05lich_merge['Weightb'] = df_p05lich_merge['Weighta'].fillna(df_p05lich_merge['Weight15'])
df_p05lich_merge['Weight'] = df_p05lich_merge['Weightb'].fillna(df_p05lich_merge['Weight16'])
df_p05lich_merge = df_p05lich_merge.drop(['Weighta','Weightb','Weight13','Weight14','Weight15','Weight16'], axis=1)

df_p05lich_merge['Irregular HRa'] = df_p05lich_merge['Irregular HR13'].fillna(df_p05lich_merge['Irregular HR14'])
df_p05lich_merge['Irregular HR'] = df_p05lich_merge['Irregular HRa'].fillna(df_p05lich_merge['Irregular HR'])
df_p05lich_merge = df_p05lich_merge.drop(['Irregular HRa','Irregular HR13','Irregular HR14'], axis=1)

df_p05lich_merge['BP sysa'] = df_p05lich_merge['BP sys13'].fillna(df_p05lich_merge['BP sys14'])
df_p05lich_merge['BP sysb'] = df_p05lich_merge['BP sysa'].fillna(df_p05lich_merge['BP sys15'])
df_p05lich_merge['BP sys'] = df_p05lich_merge['BP sysb'].fillna(df_p05lich_merge['BP sys16'])
df_p05lich_merge = df_p05lich_merge.drop(['BP sysa','BP sysb','BP sys13','BP sys14','BP sys15','BP sys16'], axis=1)

df_p05lich_merge['BP diasa'] = df_p05lich_merge['BP dias13'].fillna(df_p05lich_merge['BP dias14'])
df_p05lich_merge['BP diasb'] = df_p05lich_merge['BP diasa'].fillna(df_p05lich_merge['BP dias15'])
df_p05lich_merge['BP dias'] = df_p05lich_merge['BP diasb'].fillna(df_p05lich_merge['BP dias16'])
df_p05lich_merge = df_p05lich_merge.drop(['BP diasa','BP diasb','BP dias13','BP dias14','BP dias15','BP dias16'], axis=1)

df_p05lich_merge['Akathisiaa'] = df_p05lich_merge['Akathisia13'].fillna(df_p05lich_merge['Akathisia14'])
df_p05lich_merge['Akathisiab'] = df_p05lich_merge['Akathisiaa'].fillna(df_p05lich_merge['Akathisia15'])
df_p05lich_merge['Akathisia'] = df_p05lich_merge['Akathisiab'].fillna(df_p05lich_merge['Akathisia16'])
df_p05lich_merge = df_p05lich_merge.drop(['Akathisiaa','Akathisiab','Akathisia13','Akathisia14','Akathisia15','Akathisia16'], axis=1)

df_p05lich_merge['Acute dystoniaa'] = df_p05lich_merge['Acute dystonia13'].fillna(df_p05lich_merge['Acute dystonia14'])
df_p05lich_merge['Acute dystoniab'] = df_p05lich_merge['Acute dystoniaa'].fillna(df_p05lich_merge['Acute dystonia15'])
df_p05lich_merge['Acute dystonia'] = df_p05lich_merge['Acute dystoniab'].fillna(df_p05lich_merge['Acute dystonia16'])
df_p05lich_merge = df_p05lich_merge.drop(['Acute dystoniaa','Acute dystoniab','Acute dystonia13','Acute dystonia14','Acute dystonia15','Acute dystonia16'], axis=1)

df_p05lich_merge['Tardive dyskinesiaa'] = df_p05lich_merge['Tardive dyskinesia13'].fillna(df_p05lich_merge['Tardive dyskinesia14'])
df_p05lich_merge['Tardive dyskinesiab'] = df_p05lich_merge['Tardive dyskinesiaa'].fillna(df_p05lich_merge['Tardive dyskinesia15'])
df_p05lich_merge['Tardive dyskinesia'] = df_p05lich_merge['Tardive dyskinesiab'].fillna(df_p05lich_merge['Tardive dyskinesia16'])
df_p05lich_merge = df_p05lich_merge.drop(['Tardive dyskinesiaa','Tardive dyskinesiab','Tardive dyskinesia13','Tardive dyskinesia14','Tardive dyskinesia15','Tardive dyskinesia16'], axis=1)

df_p05lich_merge['Parkinsonisma'] = df_p05lich_merge['Parkinsonism13'].fillna(df_p05lich_merge['Parkinsonism14'])
df_p05lich_merge['Parkinsonismb'] = df_p05lich_merge['Parkinsonisma'].fillna(df_p05lich_merge['Parkinsonism15'])
df_p05lich_merge['Parkinsonism'] = df_p05lich_merge['Parkinsonismb'].fillna(df_p05lich_merge['Parkinsonism16'])
df_p05lich_merge = df_p05lich_merge.drop(['Parkinsonisma','Parkinsonismb','Parkinsonism13','Parkinsonism14','Parkinsonism15','Parkinsonism16'], axis=1)

df_p05lich = df_p05lich_merge.drop(columns={'Date13','Date14','Date15','Date16'})

#Create a file containing the relevant variables 
df_p05lich.to_csv('df_p05lich.csv')

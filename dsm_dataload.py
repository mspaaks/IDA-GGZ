import pandas as pd
import numpy as np

# Load the data
path_dsm1 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\Nagestuurde data DSM GGZ\DSM19000101-19950101.csv'
path_dsm2 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\Nagestuurde data DSM GGZ\DSM19950101-20050101.csv'
path_dsm3 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\Nagestuurde data DSM GGZ\DSM20050101-20150101.csv'
path_dsm4 = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\Nagestuurde data DSM GGZ\DSM20150101-20230101.csv'
path_baseline = r'C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\Nagestuurde data DSM GGZ\leeftijd-geslacht.csv'
path_ICD9 = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\ICDcodes_clustered.xlsx"

df_dsm1 = pd.read_csv(path_dsm1, sep=';')
df_dsm2 = pd.read_csv(path_dsm2, sep=',')
df_dsm3 = pd.read_csv(path_dsm3, sep=',')
df_dsm4 = pd.read_csv(path_dsm4, sep=',')
df_baseline = pd.read_csv(path_baseline, sep=',')
df_icd_clust = pd.read_excel(path_ICD9)

df_dsm1 = df_dsm1.rename(columns={'CLIENTNUMMER':'patient_id','ICD9_CODE':'ICD9'})
df_dsm2 = df_dsm2.rename(columns={'CLIENTNUMMER':'patient_id','ICD9_CODE':'ICD9'})
df_dsm3 = df_dsm3.rename(columns={'CLIENTNUMMER':'patient_id','ICD9_CODE':'ICD9'})
df_dsm4 = df_dsm4.rename(columns={'CLIENTNUMMER':'patient_id','ICD9_CODE':'ICD9'})
df_baseline = df_baseline.rename(columns={'CLIENT_NR':'patient_id','LEEFTIJD':'Age','GESLACHT':'Gender'})

#Transform DSM-files from long to wide format
df_dsm1_w=pd.pivot_table(df_dsm1, index='patient_id', columns = 'ICD9',values = 'DSMIV', aggfunc ='sum') #Reshape dataframe from long to wide, using ICD9 codes
df_dsm2_w=pd.pivot_table(df_dsm2, index='patient_id', columns = 'ICD9',values = 'DSMIV', aggfunc='sum') 
df_dsm3_w=pd.pivot_table(df_dsm3, index='patient_id', columns = 'ICD9',values = 'DSMIV',aggfunc='sum')
df_dsm4_w=pd.pivot_table(df_dsm4, index='patient_id', columns = 'ICD9',values = 'DSMIV',aggfunc='sum')

#Merge DSM dataframes 
df_dsm_merge = pd.concat([df_dsm1_w,df_dsm2_w, df_dsm3_w, df_dsm4_w], axis=0)

# Replace text with labels '0' if diagnosis not present and '1' if present
df_dsm_merge_n = df_dsm_merge.notnull().astype("int")

# Create function to identify all patients labeled with 'Intellectual Disability'(ID)
def label_ID (row):
   if row['317'] == 1 :
      return 1 
   if row['V62.89']==1 :
      return 1
   if row['318.0']==1 :
      return 1
   if row['318.1']==1 :
      return 1
   if row['319']==1 :
      return 1
   if row['316']==1 :
      return 1
   return 0

# Create extra column with ID labels: 1 = ID, 0 = no ID
df_dsm_merge_n['ID_label'] = df_dsm_merge_n.apply(lambda row: label_ID(row), axis=1)

# Drop columns with ID-ICD codes 
df_dsm_merge_n = df_dsm_merge_n.drop(columns={'317','V62.89','318.0','318.1','319','316'})

# Only keep the ICD-codes that are present in df_dsm_merge_n
icd_overlap = []
for element in df_icd_clust['ICD_code']:
    if element in df_dsm_merge_n.columns:
        icd_overlap.append(element)

df_icd_clust = df_icd_clust.loc[df_icd_clust['ICD_code'].isin(icd_overlap)]

#Create dict from ICD9-code clustering sheet to allow for merging on clusters
keys_groep = df_icd_clust['Groep'].unique()
dict_clusters = {}

for x in keys_groep:
   ind = df_icd_clust.index[df_icd_clust['Groep']==x].tolist()
   values = df_icd_clust['ICD_code'][ind].tolist()
   dict_clusters[x]=values


for key,values in dict_clusters.items():
   values_str = list(map(str,values))
   df_dsm_merge_n[key] = df_dsm_merge_n[values_str].sum(axis=1)

#Create files containing the relevant variables 
df_dsm = df_dsm_merge_n[['Dementia','Alcohol-induced mental disorders','Drug-induced mental disorders','Transient mental disorders due to conditions classified elsewhere','Persistent mental disorders due to conditions classified elsewhere','Schizophrenic disorders','Episodic mood disorders','Delusional disorders','Other nonorganic psychoses','Pervasive developmental disorders','Anxiety, dissociative and somatoform disorders','Personality disorders','Sexual and gender identity disorders','Alcohol dependence syndrome','Drug dependence','Nondependent abuse of drugs','Special symptoms or syndromes not elsewhere classified','Acute reaction to stress','Adjustment reaction','Specific nonpsychotic mental disorders due to brain damage','Disturbance of conduct not elsewhere classified','Disturbance of emotions specific to childhood and adolescence','Hyperkinetic syndrome of childhood','Specific delays in development','Other personal history presenting hazards to health','Other family circumstances','Other psychosocial circumstances','Other persons seeking consultation','Observation and evaluation for suspected conditions not found','ID_label']]
df_dsm.to_csv('df_dsm_all.csv')
#Correct for patients present in multiple DSM-files using groupby patient_id, get sum of diagnosis per patient
df_dsm_gr = df_dsm.groupby(['patient_id']).sum()
#Make sure the ID_label only contains values 1 or 0 (ID or no ID)
df_dsm_gr['ID_label'] = df_dsm_gr['ID_label'].replace([3], 1)
df_dsm_gr['ID_label'] = df_dsm_gr['ID_label'].replace([2], 1)
df_dsm_gr.to_csv('df_dsm_gr.csv')
df_baseline.to_csv('df_baseline.csv')
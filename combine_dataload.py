import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy import stats
from statistics import mean
from statistics import stdev

#Import created files containing relevant variables 
path_p02psyf = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_p02psyf.csv"
path_p03anam = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_p03anam.csv"
path_p04medic = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_p04medic.csv"
path_p05lich = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_p05lich.csv"
path_p06labbe = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_p06labbe.csv"
path_p09gaf = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_p09gaf.csv"
path_dsm = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_dsm_gr.csv"
path_baseline = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_baseline.csv"

df_p02psyf = pd.read_csv(path_p02psyf, sep=',')
df_p03anam = pd.read_csv(path_p03anam, sep=',')
df_p04medic = pd.read_csv(path_p04medic, sep=',')
df_p05lich = pd.read_csv(path_p05lich, sep=',')
df_p06labbe = pd.read_csv(path_p06labbe, sep=',')
df_p09gaf = pd.read_csv(path_p09gaf, sep=',')
df_dsm = pd.read_csv(path_dsm, sep=',')
df_baseline = pd.read_csv(path_baseline, sep=',')

# Merge different dataframes
df_complete = df_p02psyf.merge(df_p03anam, on='patient_id',how='outer').merge(df_p04medic, on='patient_id',how='outer').merge(df_p05lich, on='patient_id',how='outer').merge(df_p06labbe, on='patient_id',how='outer').merge(df_p09gaf, on='patient_id',how='outer')
df_complete = df_complete.merge(df_dsm, on='patient_id', how='outer') # Merge with DSM-diagnosis
# Create overview of percentage NaN's for every variable 
df_complete_nan = (df_complete.isnull().sum(axis = 0))/(len(df_complete))*100 
df_complete.to_csv('df_complete_no_baseline.csv')

# Drop columns and rows with many missing values  
thres_col = 0.50 
thres_row = 0.50

df_drop = df_complete.dropna(axis=0, thresh = thres_row*list(df_complete.shape)[1]) # drop rows
df_drop_nan = (df_drop.isnull().sum(axis = 0))/(len(df_drop))*100 
df_col = df_drop.dropna(axis=1, thresh = thres_col*list(df_drop.shape)[0]) # drop columns
df_col_nan = (df_col.isnull().sum(axis = 0))/(len(df_col))*100 
# print(list(df_complete.shape)[0])

# print(df_dsm.describe())
df_ID = df_col.loc[df_col['ID_label']== 1.0]
df_no_ID = df_col.loc[df_col['ID_label']== 0.0]

# rename ID- and no-ID dataframes 
df_ID_1 = df_ID 
df_ID_0_all = df_no_ID

# Merge the dataframes of ID and no ID with the baseline characteristics in df baseline (age and gender)
# Convert 'vrouw' to '1', 'man' to '0' and 'onbekend' to NaN or 2. Drop the NaN's. 
df_0_baseline = df_ID_0_all.merge(df_baseline, on='patient_id', how='inner')
df_1_baseline = df_ID_1.merge(df_baseline, on='patient_id', how='inner')
df_0_baseline['Gender'] = df_0_baseline['Gender'].replace(['vrouw'], '1')
df_0_baseline['Gender'] = df_0_baseline['Gender'].replace(['man'], '0')
#df_0_baseline['Gender'] = df_0_baseline['Gender'].replace(['onbekend'], np.nan)
df_0_baseline['Gender'] = df_0_baseline['Gender'].replace(['onbekend'], '2') 
df_0_baseline = df_0_baseline.dropna(subset=['Gender'])
df_0_baseline = df_0_baseline.dropna(subset=['Age'])
df_1_baseline['Gender'] = df_1_baseline['Gender'].replace(['vrouw'], '1')
df_1_baseline['Gender'] = df_1_baseline['Gender'].replace(['man'], '0')
#df_1_baseline['Gender'] = df_1_baseline['Gender'].replace(['onbekend'], np.nan)
df_1_baseline['Gender'] = df_1_baseline['Gender'].replace(['onbekend'], '2')
df_1_baseline = df_1_baseline.dropna(subset=['Gender'])
df_1_baseline = df_1_baseline.dropna(subset=['Age'])


# Check baseline characteristics of subset

def baseline(df_0_baseline, df_1_baseline):
    '''With this function, the age and gender of two dataframes will be statistically compared.
    Two separate dataframes of two groups including columns age and gender must be given as input.
    A dataframe with means and standard deviations of the two groups and a p-value indicating the difference
    between the two groups will be returned.'''
    # First calculate the means and stds of age in the two groups, rounded to two decimals
    mean_age_0 = np.round(df_0_baseline['Age'].mean(), decimals=2)
    std_age_0 = np.round(df_0_baseline['Age'].std(), decimals=2)
    mean_age_1 = np.round(df_1_baseline['Age'].mean(), decimals=2)
    std_age_1 = np.round(df_1_baseline['Age'].std(), decimals=2)
    # Next, find the percentage of females per group
    f_gender_0 = (df_0_baseline['Gender'].astype(int).sum())/len(df_0_baseline)
    f_gender_1 = (df_1_baseline['Gender'].astype(int).sum())/len(df_1_baseline)
    # Calculate the difference in gender with a Chi-square and the difference in age with a Student's t-test
    _, p_gender, _, _ = chi2_contingency(pd.crosstab(df_0_baseline['Gender'], df_1_baseline['Gender']))
    _, p_age = stats.ttest_ind(df_0_baseline['Age'].astype(int), df_1_baseline['Gender'].astype(int))
    # Combine the calculated values into a dictionary, that is converted to a dataframe for visualisation.
    dict_table = {'Amount of patients': [f'N={len(df_1_baseline)}', f'N={len(df_0_baseline)}', ' '],
                  'Age': [f'{mean_age_1} ± {std_age_1}', f'{mean_age_0} ± {std_age_0}', np.round(p_age, decimals=2)],
                  'Gender': [f'{np.round(f_gender_1*100, decimals=0)}% females (N={np.round(f_gender_1*len(df_1_baseline), decimals=0)})',
                             f'{np.round(f_gender_0*100, decimals=0)}% females (N={np.round(f_gender_0*len(df_0_baseline), decimals=0)})', np.round(p_gender, decimals=2)]}
    df_characteristics = pd.DataFrame.from_dict(dict_table, orient='index', columns=['ID group', 'no ID group', 'P-value'])
    return df_characteristics

# NB: uncomment the next line to see the baseline characteristics before balancing/stratifying
characteristics = baseline(df_0_baseline, df_1_baseline)
characteristics.to_csv('baseline_char_before_balance.csv')

# Create dataframes to use for balancing 
df_0_baseline.to_csv('df_0_baseline.csv')
df_1_baseline.to_csv('df_1_baseline.csv')
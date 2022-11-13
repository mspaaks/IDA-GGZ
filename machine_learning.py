import pandas as pd
import numpy as np 
from impute_data import impute_data
from model_pipeline import pipeline_model
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy import stats
from statistics import mean
from statistics import stdev
from statistics import variance 
from combine_dataload import baseline


# Loading the matched ID and no-ID data (based on age and gender) created in 'Matchen voor Marije.R'
path_matched = r"C:\Users\marij\OneDrive\Documenten\TM Jaar 2\TM Stage IDA\df_matched.csv"
df_matched_ag = pd.read_csv(path_matched, sep=',')
# Check baseline characteristics again after stratifying
# NB: uncomment the next lines to see the baseline characteristics after stratifying
#characteristics_stratified = baseline(df_matched_ag.loc[df_matched_ag['ID_label'] == 0.0], df_matched_ag.loc[df_matched_ag['ID_label'] == 0.0])
#characteristics_stratified.to_csv('Baseline_after_balancing.csv')

# Create a version of df_matched without variables Age and Gender and remove other irrelevant columns
df_matched = df_matched_ag.drop(['Age', 'Gender','distance','weights','subclass','...1','Unnamed: 0','...1','Unnamed: 0_x...2','Unnamed: 0_y...35','Unnamed: 0_x...59','Unnamed: 0_x...75','Unnamed: 0_y...61'], axis=1)
# print(df_matched.describe())

# Create df_decimal containing the number of decimals needed for every variable 
df_decimal = pd.DataFrame(np.zeros((1, len(df_matched.columns))))
df_decimal.columns = list(df_matched.columns)
# Define columns with 1 or more decimals needed
cols = ['Trigly_all', 'Chol_all', 'HDL_all', 'LDL_all', 'Gluc_all', 'HbA1c_all', 'Prolac_all', 'Leuko_all', 'Creat_all', 'ALAT_all', 'TSH_all']
df_decimal[cols] = df_decimal[cols].replace([0], 1)
df_decimal['Height'] = df_decimal['Height'].replace([0], 2)


# Defining empty lists needed for the loop
tprs_RF_all = []
aucs_RF_all = []
spec_RF_all = []
sens_RF_all = []
accuracy_RF_all = []
tprs_RF_sign = []
aucs_RF_sign = []
spec_RF_sign = []
sens_RF_sign = []
accuracy_RF_sign = []
perm_importances_dfs = []
sign_features_dfs = pd.DataFrame(columns = ['Features','Mean ± std ID','Mean ± std no ID'])

# Define figures
_, axis_RF_all = plt.subplots()
_, axis_RF_sign = plt.subplots()
_, axis_models = plt.subplots()

# Define data and labels
labels = df_matched['ID_label']
data = df_matched.drop(['patient_id', 'ID_label'], axis=1)
# Remove free text
data['TSH_all'] = data['TSH_all'].replace(['volgt'], np.nan)

# Evaluate data: define min, max and mean value and variance
df_minmax = pd.DataFrame(columns=['Max','Min','Mean','Variance'])
df_minmax['Max'] = data.max()
df_minmax['Min'] = data.min()
df_minmax['Mean'] = data.mean()
df_minmax['Variance'] = data.var()

# Correct deviating values
data['BMI'] = data['BMI'].replace([2160], np.nan)
data['Height'].values[data['Height'] <0.98] = np.nan
data['Prolac_all'].values[data['Prolac_all'] <10] = np.nan
# Use formula to describe all HbA1c measurements using the same unit 
for value in data['HbA1c_all']:
    if value > 11.9:
        data['HbA1c_all'] = data['HbA1c_all'].replace(value,((value*0.0915) + 2.15))
    else : continue 

# Uncomment the following lines to again evaluate min/max/mean/variance after correcting some values
#df_minmax = pd.DataFrame(columns=['Max','Min','Mean','Variance'])
#df_minmax['Max'] = data.max()
#df_minmax['Min'] = data.min()
#df_minmax['Mean'] = data.mean()
#df_minmax['Variance'] = data.var()

from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

# Create function to select significant featuers
def feature_selection(train_data, train_label, index_train, sign_features_dfs):
    '''With this function, feature selection is done using statistical testing. Dataframes with train data, train labels and train indices
    must be given as input in order to merge the train data and labels. Also lists of keys of ordinal features, binary features and continuous features
    must be given as input. A dataframe with significant features must be given as input and will be appended every fold. This appended list is returned
    and can be used for reporting. Also, a list of only the significant features in this fold is returned. This can be used for the creation of models
    with only significant features. The p-values are corrected with a Holm-Bonferroni correction.'''
    # Merge data with labels again for statistics
    merge_data_train = train_data.merge(train_label, on=index_train, how='inner')
    # Create two dataframes for the different populations
    df_num_0 = merge_data_train.loc[merge_data_train['ID_label'] == 0.0]
    df_num_1 = merge_data_train.loc[merge_data_train['ID_label'] == 1.0]
    # Create dataframe to fill with p-values
    df_p = pd.DataFrame({'Features': data.columns})

    for column_name in df_num_0.columns: 
        try:
            U1,p = mannwhitneyu(df_num_0[column_name],df_num_1[column_name])
        except: continue
        # Find significant p-values by Holm-Bonferroni:
        df_p.loc[df_p['Features'] == column_name, 'P-value'] = p   # Fill dataframe with p-values
        # Calculate the mean and std for the two populations and fill in dataframe
        mean_ID = np.round(df_num_1[column_name].mean(), decimals=2)
        std_ID = np.round(df_num_1[column_name].std(), decimals=2)
        mean_no_ID = np.round(df_num_0[column_name].mean(), decimals=2)
        std_no_ID = np.round(df_num_0[column_name].std(), decimals=2)
        df_p.loc[df_p['Features'] == column_name, 'Mean ± std ID'] = f'{mean_ID} ± {std_ID}'
        df_p.loc[df_p['Features'] == column_name, 'Mean ± std no ID'] = f'{mean_no_ID} ± {std_no_ID}'
        df_p_sorted = df_p.sort_values(by=['P-value'])    # Sort the values by p-values
        df_p_sorted['Rank'] = range(1, len(df_p_sorted)+1)    # Rank the features
        df_p_sorted['Significance level'] = 0.05/(len(df_p_sorted)+1-df_p_sorted['Rank'])    # Calculate the significance level per feature
        df_p_sorted['Significant'] = np.where(df_p_sorted['P-value'] < df_p_sorted['Significance level'], 'Yes', 'No')    # Find which features are significant
        # Create dataframe with significant features only and create table for visualisation
        df_p_sign = df_p_sorted.loc[df_p_sorted['Significant'] == 'Yes']
        df_p_for_table = df_p_sign 
        # Append the dataframe with significant features to a list for every fold. In this list, the dataframes for the 5 folds are stored.
        sign_features_dfs = sign_features_dfs.append(df_p_for_table)
        # Create list of significant features that can be used for model creation
        sign = df_p_sign['Features'].tolist()
    return sign, sign_features_dfs


# Create a pipeline for a machine learning model 
def pipeline_model(train_data, train_label, test_data, test_label, clf, tprs, aucs, spec, sens, accuracy, axis):
    '''In this function, a machine learning model is created and tested. Dataframes of the train data, train labels, test data and test labels
    must be given as input. Also, the classifier must be given as input. Scoring metrics true positives, area under curve, specificity, sensitivity
    and accuracy must be given as input, these scores are appended every fold and are returned. The axis must also be given in order to plot the ROC curves
    for the different folds in the right figure.'''
    # Fit and test the classifier
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)

    # plot ROC-curve per fold
    mean_fpr = np.linspace(0, 1, 100)    # Help for plotting the false positive rate
    viz = metrics.plot_roc_curve(clf, test_data, test_label, name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=axis)    # Plot the ROC-curve for this fold on the specified axis.
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)    # Interpolate the true positive rate
    interp_tpr[0] = 0.0    # Set the first value of the interpolated true positive rate to 0.0
    tprs.append(interp_tpr)   # Append the interpolated true positive rate to the list
    aucs.append(viz.roc_auc)    # Append the area under the curve to the list

    # Calculate the scoring metrics
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()   # Find the true negatives, false positives, false negatives and true positives from the confusion matrix
    spec.append(tn/(tn+fp))    # Append the specificity to the list
    sens.append(tp/(tp+fn))    # Append the sensitivity to the list
    accuracy.append(metrics.accuracy_score(test_label, predicted))    # Append the accuracy to the list

    return tprs, aucs, spec, sens, accuracy


# Create function to plot learning curves
def plot_learning_curve(estimator, title, X,y,axes,ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.5,1.0,5)):
        
    axes.set_title(title)
    if ylim is not None:   
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training samples")
    axes.set_ylabel("Score")
        
    train_sizes, train_scores, test_scores = \
            learning_curve(estimator, X,y, cv = cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis =1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis =1)
    test_scores_std = np.std(test_scores, axis=1)
        
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,alpha=0.1, color='r')
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,alpha=0.1, color='g')
    axes.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    axes.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross validation score')
    axes.legend(loc='best')
        
    return plt, test_scores


# Define 5-fold stratified cross-validation
cv_5fold = model_selection.StratifiedKFold(n_splits=5) 

for i, (train_index, test_index) in enumerate(cv_5fold.split(data, labels)):    # Split the data in a train and test set in a 5-fold cross-validation
    data_train = data.iloc[train_index]
    label_train = labels.iloc[train_index]
    data_test = data.iloc[test_index]
    label_test = labels.iloc[test_index]

    # Pre-processing steps
    # Impute data
    impute_train, impute_test = impute_data(data_train, data_test, df_decimal)
    # Find significant features per fold
    sign, sign_features_dfs = feature_selection(impute_train, label_train, train_index, sign_features_dfs)
    signdf = pd.DataFrame(sign)
    sign = sign[:10]
    signdf.to_excel('df_sign_features' + str(i) + '.xlsx') # Create a file every fold
    sign_features_dfs = sign_features_dfs.sort_values(by="Features") 
    sign_features_dfs = sign_features_dfs.drop_duplicates("Features", keep='first')
    sign_features_dfs = sign_features_dfs.sort_values(by="P-value") 
    sign_features_dfs = sign_features_dfs.iloc[:10]
    sign_features_dfs.to_excel('df_sign_features_meanstd' + str(i) + '.xlsx') # Create a file every fold, containing features, mean and std, p-value


    # Define classifiers
    clf_RF_all = RandomForestClassifier()
    clf_RF_sign = RandomForestClassifier()

    # Create and test two different models: random forest with all features, random forest with significant features only 
    # Random forest with all features: create model
    tprs_RF_all, aucs_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all = pipeline_model(impute_train, label_train, impute_test, label_test, clf_RF_all, tprs_RF_all, aucs_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, axis_RF_all)
    # Random forest with all features: Calculate permutation feature importance
    result = permutation_importance(clf_RF_all, impute_test, label_test, n_repeats=10, random_state=42, n_jobs=2)
    # Create dataframe to store the results
    df_feature_importance = pd.DataFrame({'Feature': (list(data_train.columns)), 'Feature importance mean': result.importances_mean, 'Feature importance std': result.importances_std})
    # Sort dataframe with the most important features first. Keep only the 5 most important features with .head()
    df_feature_importance_sorted = df_feature_importance.sort_values(by=['Feature importance mean'], ascending=False).head()
    df_feature_importance_sorted.to_excel('df_feature_importance_sorted' + str(i) + '.xlsx') # Create a file every fold
    # Append dataframe to list per fold. The list consists of i dataframes for the number of folds, showing the best 5 features per fold. This dataframe can be used for visualization.
    perm_importances_dfs.append(df_feature_importance_sorted)

    # Random forest with significant features only: create model
    tprs_RF_sign, aucs_RF_sign, spec_RF_sign, sens_RF_sign, accuracy_RF_sign = pipeline_model(impute_train[sign], label_train, impute_test[sign], label_test, clf_RF_sign, tprs_RF_sign, aucs_RF_sign, spec_RF_sign, sens_RF_sign, accuracy_RF_sign, axis_RF_sign)

    print(f'This is fold {i}')

# Plot learning curves
fig = plt.figure(figsize=(24,24))
num = 0
cvs = model_selection.StratifiedKFold(n_splits=5)
clsf = [clf_RF_all]
titles = ['RF all features']
for model in clsf:
    title = titles[num]
    ax = fig.add_subplot(2,2, num +1)
    plot_learning_curve(model, title, impute_train, label_train, ax, ylim=(0.3,1.01), cv=cvs) 
    num += 1

# Plot learning curves RF with significant features
fig = plt.figure(figsize=(24,24))
num = 0
cvs = model_selection.StratifiedKFold(n_splits=5)
clsf = [clf_RF_sign]
titles = ['RF sign features']
for model in clsf:
    title = titles[num]
    ax = fig.add_subplot(2,2, num +1)
    plot_learning_curve(model, title, impute_train[sign], label_train, ax, ylim=(0.3,1.01), cv=cvs) 
    num += 1

# Combine true positive rates, areas under curve and axes for plotting mean ROC curves
all_tprs = [tprs_RF_all, tprs_RF_sign]
all_aucs = [aucs_RF_all, aucs_RF_sign]
all_axes = [axis_RF_all, axis_RF_sign, axis_models]

# Create function to plot mean ROC-curves
def mean_ROC_curves(tprs_all, aucs_all, axis_all):
    '''With this function, the mean ROC-curves of the models over a 10-cross-validation are plot.
    The true positive rates, areas under the curve and axes where the mean ROC-curve must be plot
    are given as input for different models. The figures are filled with the mean and std ROC-curve and
    can be visualized with plt.show()
    Inputs:
        tprs_all: list with true positive rates of all models
        aucs_all: list with areas under the curve of all models
        axis_all: list of all axes
    Returns:
        plot of the mean ROC curves.
    '''
    for i, (tprs, aucs, axis) in enumerate(zip(tprs_all, aucs_all, axis_all[:2])):
        # Loop over the tprs, aucs and first three axes for the figures of the three different models.
        # Calculate means and standard deviations of true positive rate, false positive rate and area under curve
        names = ['RF all features', 'RF sign features']
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_fpr = np.linspace(0, 1, 100)
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        axis.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)   # Plot the mean ROC-curve for the corresponding model
        axis_all[2].plot(mean_fpr, mean_tpr, label=fr'Mean ROC model {(names[i])} (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)  # Plot the mean ROC-curve for the corresponding model in another figure
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)    # Set the upper value of the true positive rates
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)    # Set the upper value of the true positive rates
        axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')    # Plot the standard deviations of the ROC-curves
        axis.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f'ROC-curves model {names[i]}')    # Set axes and title
        axis.legend(loc="lower right")    # Set legend
        axis_all[2].fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2, label=r'$\pm$ 1 std. dev.')    # Plot the standard deviations of the ROC-curves in another figure
        axis_all[2].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Mean ROC-curve for the two models')    # Set axes and title
        axis_all[2].legend(loc="lower right")    # Set legend
    return

mean_ROC_curves(all_tprs,all_aucs,all_axes)

# print(sign_features_dfs)    # Print in order to show the significant features for every fold
print(perm_importances_dfs)    # Print in order to show the significant features with permuation feature importance per fold
plt.show()

# Create dictionary of all the scores for the model. Create dataframe for visualisation.
dict_scores = {'Model 1: RF with all features': [f'{np.round(mean(accuracy_RF_all), decimals=2)} ± {np.round(stdev(accuracy_RF_all), decimals=2)}',
                                                    f'{np.round(mean(sens_RF_all), decimals=2)} ± {np.round(stdev(sens_RF_all), decimals=2)}',
                                                    f'{np.round(mean(spec_RF_all), decimals=2)} ± {np.round(stdev(spec_RF_all), decimals=2)}',
                                                    f'{np.round(mean(aucs_RF_all), decimals=2)} ± {np.round(stdev(aucs_RF_all), decimals=2)}'],
            'Model 2: RF with significant features only': [f'{np.round(mean(accuracy_RF_sign), decimals=2)} ± {np.round(stdev(accuracy_RF_sign), decimals=2)}',
                                                    f'{np.round(mean(sens_RF_sign), decimals=2)} ± {np.round(stdev(sens_RF_sign), decimals=2)}',
                                                    f'{np.round(mean(spec_RF_sign), decimals=2)} ± {np.round(stdev(spec_RF_sign), decimals=2)}',
                                                    f'{np.round(mean(aucs_RF_sign), decimals=2)} ± {np.round(stdev(aucs_RF_sign), decimals=2)}']}
df_scores = pd.DataFrame.from_dict(dict_scores, orient='index', columns=['Accuracy', 'Sensitivity', 'Specificity', 'Area under ROC-curve'])

print(df_scores)
df_scores.to_excel('df_scores.xlsx')
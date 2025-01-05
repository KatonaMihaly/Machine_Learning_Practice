"""This algorithm deals with the preprocessing of the investigated data."""

# Import external libraries to utilise for the preprocessing------------------------------------------------------------

import pandas as pd
import os

# Get the absolute path of the folder-----------------------------------------------------------------------------------
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler

folder_path = os.path.dirname(os.path.abspath(__file__))

# Open the .csv file; the with function ensure that the .csv is closed after use----------------------------------------
with open(folder_path + '/initial_data/white_wine_quality.csv', 'r') as f:
    df_initial = pd.read_csv(f, sep=';')

# Checking for missing values-------------------------------------------------------------------------------------------
print(out := 'There are missing values!' if (df_initial.isna().sum() != 0).any() else 'There are no missing values.')
print()
if out == 'There are missing values!':
    print(df_initial.isna().sum())
    print()

# Checking and removing the duplicate data------------------------------------------------------------------------------
print(out := f'There are {df_initial.duplicated().sum()} duplicates!' if df_initial.duplicated().sum() != 0 else
      'There are no duplicates.')
print()
if out != 'There are no duplicates.':
    df_initial.drop_duplicates(keep='first')

# Checking for quality distribution-------------------------------------------------------------------------------------
plt.figure(figsize=(6, 4), dpi=100)
sns.countplot(data=df_initial, x='quality')
df_initial['quality'].value_counts()
plt.show()

# Oversampling underrepresented target variables------------------------------------------------------------------------
# Class imbalance in a dataset
majority_class = df_initial[df_initial['quality'] == 6]

minority_class3 = df_initial[df_initial['quality'] == 3]
minority_class4 = df_initial[df_initial['quality'] == 4]
minority_class5 = df_initial[df_initial['quality'] == 6]
minority_class7 = df_initial[df_initial['quality'] == 7]
minority_class8 = df_initial[df_initial['quality'] == 8]
minority_class9 = df_initial[df_initial['quality'] == 9]

oversampler = RandomOverSampler()

oversampled_minority3 = minority_class3.sample(n=len(majority_class), random_state=42, replace=True)
df_oversampled = pd.concat([oversampled_minority3, majority_class])
oversampled_minority4 = minority_class4.sample(n=len(majority_class), random_state=42, replace=True)
df_oversampled = pd.concat([oversampled_minority4, df_oversampled])
oversampled_minority5 = minority_class5.sample(n=len(majority_class), random_state=42, replace=True)
df_oversampled = pd.concat([oversampled_minority5, df_oversampled])
oversampled_minority7 = minority_class7.sample(n=len(majority_class), random_state=42, replace=True)
df_oversampled = pd.concat([oversampled_minority7, df_oversampled])
oversampled_minority8 = minority_class8.sample(n=len(majority_class), random_state=42, replace=True)
df_oversampled = pd.concat([oversampled_minority8, df_oversampled])
oversampled_minority9 = minority_class8.sample(n=len(majority_class), random_state=42, replace=True)
df_oversampled = pd.concat([oversampled_minority9, df_oversampled])

print(df_oversampled['quality'])

# Checking for quality distribution-------------------------------------------------------------------------------------
plt.figure(figsize=(6, 4), dpi=100)
sns.countplot(data=df_oversampled, x='quality')
df_oversampled['quality'].value_counts()
plt.show()


# Create a deepcopy of the initial dataframe to extend it with 'good' if quality > 5 else 'bad'
df_two_label = df_initial.copy(deep=True)

df_two_label['quality'] = ['good' if df_initial['quality'][i] > 5 else 'bad' for i in range(len(df_initial))]

# Create a deepcopy of the initial dataframe to extend it with:
# 'excellent' if quality > 7 else 'good' if quality > 5 else 'bad' if quality > 3 else 'poor'
df_four_label = df_initial.copy(deep=True)

df_four_label['quality'] = ['good' if df_initial['quality'][i] > 7 else
                            'good' if df_initial['quality'][i] > 5 else
                            'bad' if df_initial['quality'][i] > 3 else
                            'poor' for i in range(len(df_initial))]
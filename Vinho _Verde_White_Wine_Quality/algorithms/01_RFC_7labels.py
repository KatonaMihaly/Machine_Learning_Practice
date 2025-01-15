"""This algorithm deals with the preprocessing of the investigated data."""

# Import external libraries to utilise for the preprocessing------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# To compare oversampling to none---------------------------------------------------------------------------------------
oversample = True if input('Press "y" for oversampling and "x" for no oversampling.') == 'y' else False
both = True if input('Press "y" for both features and "x" for without both features.') == 'y' else False
if not both:
    feature_selection = True if input('Press "y" for alcohol feature and "x" for density feature.') == 'y' else False

# Get the absolute path of the folder-----------------------------------------------------------------------------------
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
plt.savefig('01_RFC_7labels_figure_01', dpi=100)
plt.show()

# Checking for correlation matrix---------------------------------------------------------------------------------------
plt.figure(figsize=(12, 8), dpi=100)
sns.heatmap(df_initial.corr(), vmin=-1, vmax=1, cmap='icefire', annot=True)
plt.savefig('01_RFC_7labels_figure_02', dpi=100)


# Selecting highly correlated features to the target variable-----------------------------------------------------------
relevant_features = df_initial.corr()['quality'][abs(df_initial.corr()['quality']) > 0.1]
print(relevant_features)
print()

# Keep relevant features----------------------------------------------------------------------------------------------
columns_to_keep = relevant_features.index.tolist()
df_relevant = df_initial[columns_to_keep]

# Check for multicollinearity between the features----------------------------------------------------------------------
plt.figure(figsize=(12, 8), dpi=100)
sns.heatmap(df_relevant.corr(), vmin=-1, vmax=1, cmap='icefire', annot=True)
plt.savefig('01_RFC_7labels_figure_03', dpi=100)
plt.show()

# Delete density as it has high correlation with alcohol----------------------------------------------------------------
if not both:
    if feature_selection:
        del df_relevant['density']
    else:
        del df_relevant['alcohol']
else:
    del df_relevant['density']
    del df_relevant['alcohol']
df_temp = df_initial.copy(deep=True)

# Oversampling underrepresented target variables------------------------------------------------------------------------
if oversample:
    majority_class = df_initial[df_initial['quality'] == 6]

    minority_class3 = df_initial[df_initial['quality'] == 3]
    minority_class4 = df_initial[df_initial['quality'] == 4]
    minority_class5 = df_initial[df_initial['quality'] == 5]
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
    oversampled_minority9 = minority_class9.sample(n=len(majority_class), random_state=42, replace=True)
    df_oversampled = pd.concat([oversampled_minority9, df_oversampled])

    df_temp = df_oversampled.copy(deep=True)

    # Checking for quality distribution---------------------------------------------------------------------------------
    plt.figure(figsize=(6, 4), dpi=100)
    sns.countplot(data=df_oversampled, x='quality')
    df_oversampled['quality'].value_counts()
    plt.savefig(f'01_RFC_7labels_figure_{"both" if both else "wboth" if not both else "alcohol" if feature_selection else "density"}', dpi=100)
    plt.show()

# Splitting data into training and testing set--------------------------------------------------------------------------
X = df_temp.drop('quality', axis=1)
Y = df_temp['quality']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Training the model ---------------------------------------------------------------------------------------------------
model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_prediction = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)-----------------------------------------------------------------------------------
mae = mean_absolute_error(Y_test, Y_prediction)

# Calculate Mean Squared Error (MSE)------------------------------------------------------------------------------------
mse = mean_squared_error(Y_test, Y_prediction)

# Calculate Root Mean Squared Error (RMSE)------------------------------------------------------------------------------
rmse = np.sqrt(mse)

# Calculate R-squared (R^2) score---------------------------------------------------------------------------------------
r2 = r2_score(Y_test, Y_prediction)

# Print the scores------------------------------------------------------------------------------------------------------
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R^2: {r2}')
print()

# Plot the scores-------------------------------------------------------------------------------------------------------
metrics = ['MAE', 'MSE', 'RMSE', 'R^2']
values = [mae, mse, rmse, r2]
plt.clf()
plt.bar(metrics, values)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel('Scores', fontsize=20)
plt.ylabel('Values', fontsize=20)
plt.gca().yaxis.grid(True, linestyle='-', alpha=0.5)
plt.savefig(f'01_RFC_7labels_figure_{"both" if both else "wboth" if not both else "alcohol" if feature_selection else "density"}_'
            f'{"oversampled" if oversample else "notoversampled"}', dpi=100)
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
"""This algorithm deals with the preprocessing of the investigated data."""

# Import external libraries to utilise for the preprocessing------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, precision_score, \
    accuracy_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

# To compare oversampling to none---------------------------------------------------------------------------------------
oversample = True if input('Press "y" for oversampling and "x" for no oversampling.') == 'y' else False

# Get the absolute path of the folder-----------------------------------------------------------------------------------
folder_path = os.path.dirname(os.path.abspath(__file__))

# Open the .csv file; the with function ensure that the .csv is closed after use----------------------------------------
with open(folder_path + '/initial_data/white_wine_quality.csv', 'r') as f:
    df_initial = pd.read_csv(f, sep=';')

# Create a deepcopy of the initial dataframe to modify it with 'good' if quality > 5 else 'bad'
df_two_label = df_initial.copy(deep=True)

df_two_label['quality'] = [1 if df_initial['quality'][i] > 5 else 0 for i in range(len(df_initial))]

# Checking for missing values-------------------------------------------------------------------------------------------
print(out := 'There are missing values!' if (df_initial.isna().sum() != 0).any() else 'There are no missing values.')
print()
if out == 'There are missing values!':
    print(df_two_label.isna().sum())
    print()

# Checking and removing the duplicate data------------------------------------------------------------------------------
print(out := f'There are {df_two_label.duplicated().sum()} duplicates!' if df_two_label.duplicated().sum() != 0 else
      'There are no duplicates.')
print()
df_two_label = df_two_label.drop_duplicates(keep='first')
df_two_label.reset_index(drop=True, inplace=True)

# Checking for quality distribution-------------------------------------------------------------------------------------
plt.figure(figsize=(6, 4), dpi=100)
sns.countplot(data=df_two_label, x='quality')
df_two_label['quality'].value_counts()
plt.savefig('02_RFC_2labels_figure_01', dpi=100)
plt.show()

# Checking for correlation matrix---------------------------------------------------------------------------------------
plt.figure(figsize=(12, 8), dpi=100)
sns.heatmap(df_two_label.corr(), vmin=-1, vmax=1, cmap='icefire', annot=True)
plt.savefig('02_RFC_7labels_figure_02', dpi=100)
plt.show()

# Selecting highly correlated features to the target variable-----------------------------------------------------------
relevant_features = df_two_label.corr()['quality'][abs(df_two_label.corr()['quality']) > 0.1]
print(relevant_features)
print()

# Keep relevant features----------------------------------------------------------------------------------------------
columns_to_keep = relevant_features.index.tolist()
df_relevant = df_two_label[columns_to_keep]

# Check for multicollinearity between the features----------------------------------------------------------------------
plt.figure(figsize=(12, 8), dpi=100)
sns.heatmap(df_relevant.corr(), vmin=-1, vmax=1, cmap='icefire', annot=True)
plt.savefig('02_RFC_7labels_figure_03', dpi=100)
plt.show()

# Oversampling underrepresented target variables------------------------------------------------------------------------
if oversample:
    majority_class = df_relevant[df_relevant['quality'] == 1]

    minority_class = df_relevant[df_relevant['quality'] == 0]

    oversampler = RandomOverSampler()

    oversampled_minority = minority_class.sample(n=len(majority_class), random_state=42, replace=True)
    df_oversampled = pd.concat([oversampled_minority, majority_class])

    df_temp = df_oversampled.copy(deep=True)

    # Checking for quality distribution---------------------------------------------------------------------------------
    plt.figure(figsize=(6, 4), dpi=100)
    sns.countplot(data=df_oversampled, x='quality')
    df_oversampled['quality'].value_counts()
    plt.savefig(f'02_RFC_7labels_figure_04', dpi=100)
    plt.show()

    # Splitting data into training and testing set----------------------------------------------------------------------
    X = df_temp.drop('quality', axis=1)
    Y = df_temp['quality']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Training the model------------------------------------------------------------------------------------------------
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    Y_prediction = model.predict(X_test)

    # Create a confusion matrix-----------------------------------------------------------------------------------------
    cm = confusion_matrix(Y_test, Y_prediction)
    cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                         columns=['Predicted Negative', 'Predicted Positive'])
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 18})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'02_RFC_7labels_figure_05', dpi=100)
    plt.show()

    precision = precision_score(Y_test, Y_prediction)
    accuracy = accuracy_score(Y_test, Y_prediction)
    recall = recall_score(Y_test, Y_prediction)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Recall (Sensitivity):", recall)

    cv = []
    precision1 = []
    precision2 = []
    accuracy1 = []
    accuracy2 = []
    sensitivity1 = []
    sensitivity2 = []

    for i in range(2, 21, 1):
        scores = cross_val_score(model, X, Y, cv=i, scoring="precision", )
        precision1.append(scores.mean())
        precision2.append(scores.std())
        scores = cross_val_score(model, X, Y, cv=i, scoring="accuracy")
        accuracy1.append(scores.mean())
        accuracy2.append(scores.std())
        scores = cross_val_score(model, X, Y, cv=i, scoring="recall")
        sensitivity1.append(scores.mean())
        sensitivity2.append(scores.std())
        cv.append(i)

    data1 = pd.DataFrame(columns=['CV', 'Precision', 'Accuracy', 'Sensitivity'])
    data1['CV'] = cv
    data1['Precision'] = precision1
    data1['Accuracy'] = accuracy1
    data1['Sensitivity'] = sensitivity1

    data2 = pd.DataFrame(columns=['CV', 'Precision', 'Accuracy', 'Sensitivity'])
    data2['CV'] = cv
    data2['Precision'] = precision2
    data2['Accuracy'] = accuracy2
    data2['Sensitivity'] = sensitivity2

    # Plot each column as a scatter plot
    plt.figure(figsize=(8, 6))
    for column in ['Precision', 'Accuracy', 'Sensitivity']:
        plt.scatter(data1['CV'], data1[column], label=column)

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('MEAN')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'02_RFC_7labels_figure_06', dpi=100)
    plt.show()

    # Plot each column as a scatter plot
    plt.figure(figsize=(8, 6))
    for column in ['Precision', 'Accuracy', 'Sensitivity']:
        plt.scatter(data2['CV'], data2[column], label=column)

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('STD')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'02_RFC_7labels_figure_07', dpi=100)
    plt.show()

else:

    df_temp = df_relevant.copy(deep=True)

    # Splitting data into training and testing set--------------------------------------------------------------------------
    X = df_temp.drop('quality', axis=1)
    Y = df_temp['quality']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Training the model ---------------------------------------------------------------------------------------------------
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    Y_prediction = model.predict(X_test)

    # Create a confusion matrix-----------------------------------------------------------------------------------------
    cm = confusion_matrix(Y_test, Y_prediction)
    cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                         columns=['Predicted Negative', 'Predicted Positive'])
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 18})
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'02_RFC_7labels_figure_08', dpi=100)
    plt.show()

    precision = precision_score(Y_test, Y_prediction)
    accuracy = accuracy_score(Y_test, Y_prediction)
    recall = recall_score(Y_test, Y_prediction)
    print("Precision:", precision)
    print("Accuracy:", accuracy)
    print("Recall (Sensitivity):", recall)

    cv = []
    precision1 = []
    precision2 = []
    accuracy1 = []
    accuracy2 = []
    sensitivity1 = []
    sensitivity2 = []

    for i in range(2, 21, 1):
        scores = cross_val_score(model, X, Y, cv=i, scoring="precision", )
        precision1.append(scores.mean())
        precision2.append(scores.std())
        scores = cross_val_score(model, X, Y, cv=i, scoring="accuracy")
        accuracy1.append(scores.mean())
        accuracy2.append(scores.std())
        scores = cross_val_score(model, X, Y, cv=i, scoring="recall")
        sensitivity1.append(scores.mean())
        sensitivity2.append(scores.std())
        cv.append(i)

    data1 = pd.DataFrame(columns=['CV', 'Precision', 'Accuracy', 'Sensitivity'])
    data1['CV'] = cv
    data1['Precision'] = precision1
    data1['Accuracy'] = accuracy1
    data1['Sensitivity'] = sensitivity1

    data2 = pd.DataFrame(columns=['CV', 'Precision', 'Accuracy', 'Sensitivity'])
    data2['CV'] = cv
    data2['Precision'] = precision2
    data2['Accuracy'] = accuracy2
    data2['Sensitivity'] = sensitivity2

    # Plot each column as a scatter plot
    plt.figure(figsize=(8, 6))
    for column in ['Precision', 'Accuracy', 'Sensitivity']:
        plt.scatter(data1['CV'], data1[column], label=column)

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('MEAN')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'02_RFC_7labels_figure_09', dpi=100)
    plt.show()

    # Plot each column as a scatter plot
    plt.figure(figsize=(8, 6))
    for column in ['Precision', 'Accuracy', 'Sensitivity']:
        plt.scatter(data2['CV'], data2[column], label=column)

    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('STD')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'02_RFC_7labels_figure_10', dpi=100)
    plt.show()

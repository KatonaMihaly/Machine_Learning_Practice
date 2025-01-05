# Importing the modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score

# Importing the dataset
df = pd.read_csv("../whitewine/winequality-white.csv", sep=";")

# Checking for missing values
print(df.isna().sum())

# Removing the duplicate data
df.drop_duplicates(keep='first')

# Checking Wine Quality Distribution
plt.figure(figsize=(6,4), dpi=100)
sns.countplot(data=df, x='quality')
# plt.show()

# Correlation matrix
plt.figure(figsize=(12,8), dpi=100)
sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap='icefire',annot=True)
# plt.show()

# Selecting highly correlated features
relevant_features= abs(df.corr()['quality'])[abs(df.corr()['quality'])>0.1]
print(relevant_features.nlargest(10))

# Keep relevant attributes
columns_to_keep = relevant_features.index.tolist()
df = df[columns_to_keep]

# Modifying the table to use Good (1) and Bad (0) markers.
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6.5 else 0)

# Checking for data imbalance
plt.figure(figsize=(6, 4), dpi=100)
sns.countplot(data=df, x='quality')
plt.xticks([0,1], ['Bad wine', 'Good wine'])
quality_counts = df['quality'].value_counts()
print(quality_counts)

# Class imbalance in a dataset
majority_class = df[df['quality'] == 0]
minority_class = df[df['quality'] == 1]
undersampled_majority = majority_class.sample(n=len(minority_class), random_state=42)
df = pd.concat([undersampled_majority, minority_class])
df = df.sample(frac=1, random_state=42)
quality_counts = df['quality'].value_counts()
print(quality_counts)

plt.figure(figsize=(10, 8),dpi=150)
for i, j in enumerate(df.drop('quality', axis=1).columns):
    plt.subplot(4, 2, i+1)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    sns.histplot(data=df, x=df[f"{j}"], hue="quality", kde=True, bins=20, multiple="stack", alpha=.2)
    plt.legend(['Good wine', 'Bad wine'])
# plt.savefig('whitewine.png')

# Splitting Data into Training and Testing Set
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'],
                     columns=['Predicted Negative', 'Predicted Positive'])
plt.figure(figsize=(8, 6))
heatmap = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 18})
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('whitewineconf.png')

precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision:", precision)
print("Accuracy:", accuracy)
print("Recall (Sensitivity):", recall)

# K-fold cross-validation on the same DataFrame


# scores = cross_val_score(model, X, y, cv=10, scoring="accuracy")
# print("Mean Accuracy:", scores.mean())
# print("Standard Deviation:", scores.std())
#
# scores = cross_val_score(model, X, y, cv=10, scoring="recall")
# print("Mean Accuracy:", scores.mean())
# print("Standard Deviation:", scores.std())

cv = []
precision1 = []
precision2 = []
accuracy1 = []
accuracy2 = []
sensitivity1 = []
sensitivity2 = []

for i in range(2, 21, 1):
    scores = cross_val_score(model, X, y, cv=i, scoring="precision",)
    precision1.append(scores.mean())
    precision2.append(scores.std())
    scores = cross_val_score(model, X, y, cv=i, scoring="accuracy")
    accuracy1.append(scores.mean())
    accuracy2.append(scores.std())
    scores = cross_val_score(model, X, y, cv=i, scoring="recall")
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

data1.to_excel('whitewine1.xlsx', index=False)
data2.to_excel('whitewine2.xlsx', index=False)
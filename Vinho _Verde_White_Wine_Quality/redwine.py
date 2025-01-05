# Importing the modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score, LeavePOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

# Importing the dataset
df = pd.read_csv("../redwine/winequality-red.csv", sep=";")

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

# Class imbalance in a dataset
majority_class = df[df['quality'] == 5]
minority_class3 = df[df['quality'] == 3]
minority_class4 = df[df['quality'] == 4]
minority_class6 = df[df['quality'] == 6]
minority_class7 = df[df['quality'] == 7]
minority_class8 = df[df['quality'] == 8]
oversampler = RandomOverSampler()
oversampled_minority3 = minority_class3.sample(n=len(majority_class), random_state=42, replace=True)
df = pd.concat([oversampled_minority3, majority_class])
oversampled_minority4 = minority_class4.sample(n=len(majority_class), random_state=42, replace=True)
df = pd.concat([oversampled_minority4, df])
oversampled_minority6 = minority_class6.sample(n=len(majority_class), random_state=42, replace=True)
df = pd.concat([oversampled_minority6, df])
oversampled_minority7 = minority_class7.sample(n=len(majority_class), random_state=42, replace=True)
df = pd.concat([oversampled_minority7, df])
oversampled_minority8 = minority_class8.sample(n=len(majority_class), random_state=42, replace=True)
df = pd.concat([oversampled_minority8, df])
df = df.sample(frac=1, random_state=42)
quality_counts = df['quality'].value_counts()
print(quality_counts)

# Splitting Data into Training and Testing Set
X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
# Calculate R-squared (R^2) score
r2 = r2_score(y_test, y_pred)

metrics = ['MAE', 'MSE', 'RMSE', 'R^2']
values = [mae, mse, rmse, r2]

plt.clf()
plt.bar(metrics, values)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.xlabel('Mérőszámok', fontsize=20)
plt.ylabel('Értékek', fontsize=20)
plt.gca().yaxis.grid(True, linestyle='-', alpha=0.5)
plt.savefig('redmetrics.png')

# Prepare your data
X = df.drop('quality', axis=1)
y = df['quality']
regressor = DecisionTreeRegressor()

data = {'Metric': ['MAE', 'MSE', 'RMSE', 'R^2'], '': ['', '', '', '']}
df = pd.DataFrame(data)

for i in [10, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]:
    n_iterations = i # Number of bootstrap iterations
    performance_values = []

    for _ in range(n_iterations):
        X_bootstrap, y_bootstrap = resample(X, y, random_state=42)
        regressor.fit(X_bootstrap, y_bootstrap)
        y_pred = regressor.predict(X)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y, y_pred)
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y, y_pred)
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        # Calculate R-squared (R^2) score
        r2 = r2_score(y, y_pred)
        performance_values.append((mae, mse, rmse, r2))
        #
    mae_values, mse_values, rmse_values, r2_values = zip(*performance_values)  # Unzip the performance values
    mean_mae = np.mean(mae_values)
    mean_mse = np.mean(mse_values)
    mean_rmse = np.mean(rmse_values)
    mean_r2 = np.mean(r2_values)

    data = {'Metric': ['MAE', 'MSE', 'RMSE', 'R^2'], str(i): [mae, mse, rmse, r2]}
    performance_df = pd.DataFrame(data)
    df = pd.concat([df, performance_df], axis=1)

columns_to_delete = [col for col in df.columns if col.startswith('Metric')]
df.drop(columns=columns_to_delete[0:], inplace=True)
df.to_excel('performance_metrics.xlsx', index=False)

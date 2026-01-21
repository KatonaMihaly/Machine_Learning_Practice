# 🍷 Machine Learning Practice: Vinho Verde Wine Quality

This repository contains practice sessions focused on implementing **Machine Learning algorithms** to solve classification problems. The primary project currently featured is the **Vinho Verde White Wine Quality Classification** using a **Random Forest Classifier**.

The project explores two distinct approaches:
* **7-Label Classification:** Predicting the exact quality score.
* **2-Label Classification:** Simplifying the target into "Good" (quality > 5) vs. "Bad" (quality <= 5).

---

## 📝 Project Overview
The goal of this project is to model wine quality based on physicochemical tests. A significant portion of the work is dedicated to exploratory data analysis (EDA), handling class imbalances, and feature engineering to optimize model performance.

### ✨ Key Features
* **Dynamic Data Handling:** Users can toggle oversampling on or off at runtime to compare its impact on model metrics.
* **Correlation Analysis:** Identification of high-impact features (e.g., alcohol content) and removal of redundant features to reduce model complexity.
* **Multicollinearity Management:** Analysis of the relationship between density and alcohol to prevent overfitting and improve interpretability.
* **Comprehensive Evaluation:** Models are assessed using Accuracy, Precision, Recall, F1-Score, and Specificity, alongside error metrics like MAE and RMSE.

---

## 🛠 Technical Stack
* **Language:** Python 🐍
* **Data Processing:** pandas, numpy
* **Visualization:** seaborn, matplotlib
* **Machine Learning:** scikit-learn (Random Forest, Metrics, Model Selection)
* **Imbalance Handling:** imblearn (Oversampling techniques)

---

## 🔄 Workflow

### 1. Preprocessing and Cleaning
* **Missing Values:** Checks for null entries and removes them.
* **Duplicate Removal:** Identifies and drops duplicate entries to prevent training bias and artificial inflation of metrics.

### 2. Exploratory Data Analysis (EDA)
* **Distribution Analysis:** Visualizes the target variable to identify class imbalances.
* **Heatmaps:** Utilizes correlation matrices to filter features with low predictive power (correlation < |0.1|).



### 3. Feature Engineering
* **Addressing Multicollinearity:** The project specifically examines the high correlation between density and alcohol. Testing shows that removing one or both can decrease model complexity without significantly impacting precision.

### 4. Model Training and Comparison
* **Oversampling Impact:** The study demonstrates that oversampling minority classes significantly improves **Specificity** (the ability to correctly identify negative labels) and overall **Accuracy** in imbalanced sets.
* **K-Fold Cross-Validation:** Ensures the model generalizes well to unseen data.



---

## 📊 Key Results
* **Oversampling Effectiveness:** In the 2-label classification, oversampling improved Accuracy by approximately **14%** and Specificity by over **35%**.
* **Feature Redundancy:** Removing the "density" feature (highly correlated with alcohol) proved effective in simplifying the model while maintaining high performance.
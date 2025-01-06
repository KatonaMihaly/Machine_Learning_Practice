Documentation of Vinho Verde White Wine Quality with Random Forest Classifier of 7 lables
-

Imported packages:
-
- numpy >> for mathematical functions, similar to math package
- pandas >> for data processing
- os >> for file path handling
- seaborn >> for plotting the data
- matplotlib >> for plotting the data
- imblearn >> for oversampling of the dataset
- sklearn >> for the machine learning algorithm

To compare oversampling to no oversampling.
-
oversample = True if input('Press "y" for oversampling and "x" for no oversampling.') == 'y' else False

_This line asks the user for an input of "y" meaning yes or "x" meaning no whether to use oversampling or not.
The goal is to compare the effect of oversampling of the underrepresented target variable values._


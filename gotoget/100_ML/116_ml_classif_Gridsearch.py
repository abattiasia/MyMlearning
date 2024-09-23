
# 116_ml_classif_Gridsearch.py
  
Apply_Gridsearch_to_define_best_model_params:
Ex.01_apply_to_ RandomForestClassifier:
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
## Assuming you have already split your data into X_train, X_test, y_train, and y_test
## Replace 'X_train', 'X_test', 'y_train', and 'y_test' with your actual data
## Define the parameter grid
param_grid = {
	'n_estimators': [30],  ## Set the desired number of estimators
	'criterion': ['entropy']  ## Set the criterion to 'entropy'

## Initialize GridSearchCV with RandomForestClassifier
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
## Fit the model to the training data
grid_search.fit(X_train, y_train)
## Evaluate the model on the test data
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
## Create a DataFrame with the results
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df)
##++++++++++++++++++++++++++++++++++
##Ex.02
param_grid = 
'n_estimators': [50, 100, 150],
'max_depth': [None, 10, 20, 30],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4],
'bootstrap': [True, False]}
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
						   param_grid=param_grid,
						   cv=5,  
						   n_jobs=-1)
grid_search.fit(X_train, y_train)
##+
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
##+
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Parameters: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
Ex.02_apply_to_KNeighborsClassifier:
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
## Assuming you have already split your data into X_train, X_test, y_train, and y_test
## Replace 'X_train', 'X_test', 'y_train', and 'y_test' with your actual data
## Define the parameter grid
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
## Initialize GridSearchCV with KNeighborsClassifier
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
## Fit the model to the training data
grid_search.fit(X_train, y_train)
## Evaluate the model on the test data
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
## Create a DataFrame with the results
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df)


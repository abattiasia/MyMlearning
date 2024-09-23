
# 114_ml_classif_Naive_Bayes_classifier_Model=GaussianNB.py

##!pip install scikit-learn

#00 define x,y and X_train , X_test for ML Model:
target = 'diabetes'
x= df_encoded.drop(columns=[target])  
y = df_encoded[target]
##+++++++++

x = df.drop(columns=['Class'], inplace=False) 
y = df['Class']

## data split 
X_train , X_test , y_train , y_test = train_test_split(x,y,test_size=.20)
##++++++++++++

#05 Classification report:
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
##++++++++
    
##the test data, and evaluates the model. If your data is categorical, 
##you may need to use CategoricalNB instead of GaussianNB.
##!pip install scikit-learn
## Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
## Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
## Initialize the Naive Bayes classifier
nb = GaussianNB()
## Fit the model on the training data
nb.fit(X_train, y_train)
## Predict on the test data
y_pred = nb.predict(X_test)
## Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
## Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

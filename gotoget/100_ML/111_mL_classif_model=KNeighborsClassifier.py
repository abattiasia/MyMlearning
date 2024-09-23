
# 110_ML_Class_model=KNeighborsClassifier.py  
    
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

#06 Comparison  between the 3 Models:
##KNeighborsClassifier() accuracy =  92%
##DecisionTreeClassifier() accuracy = 81.5%
##RandomForestClassifier()  accuracy = 82.57%
##++++++++
  
#01 KNN Prject  K Neighbors Classifier  -> model=KNeighborsClassifier():
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
## build model 
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train , y_train)
print('ok')

## predict +++++++++++++++++++
y_pred = model.predict(X_test)
y_pred

## accuracy +++++++++++++++++
from sklearn.metrics import accuracy_score
accuracy_score(y_test , y_pred)
##+++++++++++++++++++++++++++++++++++++

## choose best k 
## Define the range of k values to test
k_values = range(1,50,2)
test_accuracies = []

## Train the k-NN classifier with different values of k and evaluate on the test set
for k in k_values:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)  ## Train the model
	y_pred = knn.predict(X_test)  ## Predict on the test set
	accuracy = accuracy_score(y_test, y_pred)  ## Calculate accuracy
	test_accuracies.append(accuracy)
## Determine the best k
best_k = k_values[np.argmax(test_accuracies)]
print(f"The best k value is: {best_k}")
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## Plot the k values vs. accuracy on the test set
plt.figure(figsize=(10, 6))
plt.plot(k_values, test_accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Test Set Accuracy')
plt.title('k-NN Varying k Value')
  plt.show()
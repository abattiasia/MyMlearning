
# 112_mL_class_model=DecisionTreeClassifier.py
    
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

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
## predict and accuracy
y_pred = model.predict(X_test)
y_pred
accuracy_score(y_test,y_pred)
##+++++++++++++++++++++++++++++++++++
from sklearn import tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model)
  plt.show()
#


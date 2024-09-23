
# 113_ml_classif_model=RandomForestClassifier.py

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

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='entropy',n_estimators=30) 
model.fit(X_train,y_train)
## predict and accuracy
y_pred = model.predict(X_test)
y_pred
accuracy_score(y_test,y_pred)
## Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
## Print feature importance
print("Feature importances:")
for i in indices:
	print(f"{x.columns[i]}: {importances[i]}")
## Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices], align="center")
plt.xticks(range(x.shape[1]), np.array(x.columns)[indices], rotation=45)
plt.xlim([-1, x.shape[1]])
plt.show()

#101_ml_Regression.py

model = LinearRegression()
model.fit(X_train , y_train)
model.coef_
model.intercept_
y_pred = model.predict(X_test)
y_pred
### Model Evaluation
from sklearn.metrics import r2_score
r2_score(y_test , y_pred)



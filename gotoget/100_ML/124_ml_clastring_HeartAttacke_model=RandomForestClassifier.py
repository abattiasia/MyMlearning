

# 124_ml_clastring_HeartAttacke_model=RandomForestClassifier.py
# 100_ml_Classif_coutch-01_13092024:

print(f""" ## outline
Outline of Project: 
- import libraries 
- read data --> notes
- data explore (head ,info , describe) --> notes
- data cleaning (outliers , duplicated , column types ,null)
- visualization --> notes
- feature extraction (optional)
- split x , y
- preprocessing if needed (encoding,scallng)
- split train , test 
- build model (many algorithms)
- check accuracy 
- hyperparameter (Grid search , Cross validation , ....) --> Build model again 
- save model """)
##+++++++++
import libs:
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as xp
##++++++
load dataset: -> Heart Attacke
df = pd.read_csv('heart.csv')
df.head()
print(f""" ## Ml Notes: 
- Encoding : No (no category text columns), 
- scaling : No (values closer to each other)""")
df.info_describe
df.info()
## Notes: - no null , - no data conversion 
df.describe()
##+++++++++++++++++++++

df[cols].value_counts().count() - to get cat cols < 10
value_counts = []  ## this code to show how many repeated values in each column 
for c in df.columns:
	d = df[c].value_counts().count()
	value_counts.append(int(d))
pd.DataFrame(value_counts,index=df.columns)
##++++++++++++++++++++++++++

num_cols = ['age','trtbps','chol','thalachh','oldpeak']
cat_cols = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
##++++++++++++++++
df.duplicated()_drop
df.duplicated().sum()          ## check duplicate
df.drop_duplicates()
num_cols, cat_cols  ## to  print them
##+++++++++++++++++++++
data Visualize 
cat_cols = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
sns.countplot(x='sex',data=df)
##++++++++++++++
for c in cat_cols:
   sns.countplot(x=c,data=df,palette='Set2',hue='output')
	plt.title(f'Count {c}')
	plt.show()
##+++++++++++++++++++++++++++++
	
Model Building
df.sample(10)
df.dtypes
Print(f"""- Notes :
- no encoding (all columns are numbers)
- no scaling (number are close)
- all column types are ok """)

## importing
from sklearn.model_selection import train_test_split , GridSearchCV , cross_val_score
from sklearn.metrics import accuracy_score

## hide warnings
import warnings
warnings.filterwarnings('ignore')
##+++++++++++

## split x , y
x = df.drop(["output"],axis=1) ## get all columns except output
y = df["output"] ## get output only
print(f""" ## Build Different Models 
--> Lazypredict --> top 5 models --> each one --> hyper parameter  
""")
##+++++++++++++++++++++++++++++++++++++++++

02 Build Random Forest Modell
from sklearn.ensemble import RandomForestClassifier
## Buid model 
rf_model = RandomForestClassifier()
## train model
rf_model.fit(X_train,y_train)
## Predict
y_pred = rf_model.predict(X_test)
## accuracy Score
accuracy_score(y_test,y_pred)
##++++++++++++++++++++++++++++++++

02.01 Random Forest Modell with  grid search --> params 
## Buid model 
rf2_model = RandomForestClassifier()
## model params
params = 
	'n_estimators' : [10,25,50,100,150,200],
	'criterion' : ['entropy' , 'gini' , 'log_loss']
## build model 
rf2_gs_model = GridSearchCV(rf2_model,param_grid=params)
## train 
rf2_gs_model.fit(X_train,y_train)
##++
## best parmas
rf2_gs_model.best_params_
##++++

02.02 build random forest with best parms
rf3_model = RandomForestClassifier(criterion='entropy', n_estimators=200)
## train model
rf3_model.fit(X_train,y_train)
## Predict
y_pred = rf3_model.predict(X_test)
## accuracy Score
accuracy_score(y_test,y_pred)
##========================================

Save final Model and load it by pickle
## Save model
import pickle
model_name = 'final_model.sav'
save_model = pickle.dump(rf3_model,open(model_name,'wb'))
## Load Saved model
my_model = pickle.load(open('final_model.sav','rb'))
my_model.predict(X_test)
print('finnished')


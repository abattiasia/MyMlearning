
# 122_ml_clastring_Heart Attacke_model=DecisionTreeClassifier.py

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

01 Build Decision Tree Modell
from sklearn.tree import DecisionTreeClassifier
## Buid model 
dt_model = DecisionTreeClassifier()
## train model
dt_model.fit(X_train,y_train)
## Predict
y_pred = dt_model.predict(X_test)
## accuracy Score
accuracy_score(y_test,y_pred)
 ##+++++++++++++++++++++++++++++++++++++++++++

01.01 Decision Tree with Cross Validation 
score = cross_val_score(dt_model,X_test,y_test,cv =5)  ## cv = 5 --> fold
score.mean()
##+++++++++++++++++++
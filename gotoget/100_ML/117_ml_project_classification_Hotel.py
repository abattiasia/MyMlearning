
# 117_ml_project_classification_Hotel.py
## Classification_Hotel.pyv  10.09.2024
## Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from plotly import tools
##import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
##init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

##import xgboost as xgb
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, RocCurveDisplay, ConfusionMatrixDisplay


##MAIN_PATH = '../input/'
df = pd.read_csv(r'hotel_booking.csv')
##term_deposits = df.copy()
## Have a grasp of how our data looks.
df.head()

df.describe()  3 df.describe


## No missing values.
df.info()

## Bivariate bar plot of 'is_canceled' vs 'reservation_status' with specified colors
plt.figure(figsize=(10, 5))
sns.countplot(x='reservation_status', hue='is_canceled', data=df, palette=['darkturquoise', 'royalblue'])
plt.title('Count of Cancellations by Reservation Status')
plt.xlabel('Reservation Status')
plt.ylabel('Count')
plt.legend(title='Is Canceled', labels=['Not Canceled', 'Canceled'])
plt.show()

##====================================
## to see dataset
fig = px.scatter(df['total_of_special_requests'],df['is_canceled'])
fig.show()
##==============================================
df.isnull().count()
df.isnull().sum()

df.drop(columns=['company'], inplace=True)  ##df= df.drop(columns=['company'], inplace=True)
df.isnull().sum()

##df.fillna(df.mean('agent'), inplace=True)  ## df1= df.fillna(df.mean(), inplace=True)  is not working

##by Number columns can be fill null values with mean() so:
df.fillna({ 'agent': df['agent'].mean()}, inplace=True)
df.isnull().sum()

##by Object columns nust be fill nul values with mode() so:
df.fillna({ 'country': df['country'].mode()[0] }, inplace=True)
df.isnull().sum()
##by Object columns nust be fill nul values with mode() so:
df.fillna({ 'children': df['children'].mode()[0] }, inplace=True)
df.isnull().sum()

##====================
## Calculating the percentage of each class
percentage = df['is_canceled'].value_counts(normalize=True) * 100

## Plotting the percentage of each class
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=percentage.index, y=percentage, palette=['darkturquoise', 'royalblue'])
plt.title('Percentage of Cancellations and Non-Cancellations')
plt.xlabel('Is Canceled')
plt.ylabel('Percentage (%)')
plt.xticks(ticks=[0, 1], labels=['Not Canceled', 'Canceled'])
plt.yticks(ticks=range(0,80,10))

## Displaying the percentage on the bars
for i, p in enumerate(percentage):
	ax.text(i, p + 0.5, f'{p:.2f}%', ha='center', va='bottom')

plt.show()

##=========================================
##alle not important columns will be droped from db befor startin with ML Model
df.drop(columns=['email', 'phone-number', 'credit_card', 'credit_card', 'reservation_status_date', 'reservation_status', 'customer_type',], inplace=True)
##===========================
df.info()
##=======

## make a list for numerical_columns and object_columns
numerical_columns = df.select_dtypes(include=['number']).columns
object_columns = df.select_dtypes(include=['object']).columns
##++++++++++++++++++++++++++++++++++++++
num_columns = df.select_dtypes(include=['number']).columns
num_columns

## ++++++++++++++++++++++++++++++++++
## try to understand how make a list for numerical_columns 

numeric_cols= list(df.select_dtypes(exclude=['object']).columns)
cat_cols=['hotel', 'meal', 'market_segment', 'distribution_channel', 'customer_type', 'country']
ordinal_cols=['reserved_room_type','assigned_room_type','deposit_type','arrival_date_month', 'reservation_status_date'] 

data=df[[*cat_cols, *numeric_cols]]  
 
data = pd.get_dummies(data, columns=cat_cols)
new_columns={}
for c in data.columns:
	new_columns[c]=c.replace(' ', '_')
data.rename(columns=new_columns, inplace=True)

#+++++++++++++++++++++++++++++++++++++++++++++++++++

##=========================================
## Classification  KNN, Naive Bayes
## 01 KNN Build Model
##++++++++++++++++++++++++++

from sklearn.preprocessing import LabelEncoder

## Initialize the encoder
le = LabelEncoder()

## Apply label encoding

df['hotel'] = le.fit_transform(df['hotel'])
df['arrival_date_month'] = le.fit_transform(df['arrival_date_month'])
df['meal'] = le.fit_transform(df['meal'])
df['country'] = le.fit_transform(df['country'])
df['market_segment'] = le.fit_transform(df['market_segment'])
df['distribution_channel'] = le.fit_transform(df['distribution_channel'])
df['reserved_room_type'] = le.fit_transform(df['reserved_room_type'])
df['assigned_room_type'] = le.fit_transform(df['assigned_room_type'])
df['name'] = le.fit_transform(df['name'])
df['deposit_type'] = le.fit_transform(df['deposit_type'])
df
##++++++++++++++++++++++++++
## define x and y
x = df.drop(columns=['is_canceled'], inplace=False) 
y = df['is_canceled']
##++++++++++++++++++++++++++
## data split 
X_train , X_test , y_train , y_test = train_test_split(x,y,test_size=.20)
##+++++++++++++++++++++
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
## build model 
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train , y_train)
print('ok')
##++++++++++++++++++++++
## predict 
y_pred = model.predict(X_test)
y_pred
##+++++++++++++++++++++++++++++++++++
## accuracy 
from sklearn.metrics import accuracy_score
accuracy_score(y_test , y_pred)
##+++++++++++++++++++++++++++++++++++
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
##++++++++++++++++++++++++++++++++++++++++
## Plot the k values vs. accuracy on the test set
plt.figure(figsize=(10, 6))
plt.plot(k_values, test_accuracies, marker='o')
plt.xlabel('k')
plt.ylabel('Test Set Accuracy')
plt.title('k-NN Varying k Value')
plt.show()
##==============================================================

## 02 Naive Bayes classifier  Build Model
##This code splits the data, trains the Naive Bayes classifier, predicts 
##the test data, and evaluates the model. If your data is categorical, 
##you may need to use CategoricalNB instead of GaussianNB.

##!pip install scikit-learn
## Import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

## Assuming x and y are already defined as your feature matrix and target variable
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

## Classification report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)


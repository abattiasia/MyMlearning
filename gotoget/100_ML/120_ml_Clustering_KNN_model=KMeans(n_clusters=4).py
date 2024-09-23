

#120_ml_Clustering_KNN_model=KMeans(n_clusters=4).py
00 prepear_dataset:
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
##Clustering Exercises EDA¶
##Clustering Using Methods You Know
## read data 
df = pd.read_csv('/kaggle/input/clustering-exercises/basic1.csv')
df.describe()  ##.T
df.head()
##======================
x = df[['x','y']]
y_labels=df['color']
plt.scatter(x['x'],x['y'],c=y_labels,cmap='rainbow')

01 KNN Model KMean Clustering -> model=KMeans():{
model = KMeans(n_clusters=4)
model.fit(x)
y_labels = model.predict(x)
## test predict
model.predict([[15,70]])
plt.scatter(x['x'],x['y'],c=y_labels,cmap='rainbow')
}
02 Agglomerative Hierarchical Clustering -> model = AgglomerativeClustering():{
from sklearn.cluster import AgglomerativeClustering
##++++++++++++
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(12,8))
dg = sch.dendrogram(sch.linkage(x,method='average'))
##++++++++++++
## build model
model = AgglomerativeClustering(n_clusters=4)
model.fit(x)
y_lables = model.fit_predict(x)
y_lables
plt.scatter(x['x'],x['y'],c=y_lables,cmap='rainbow')
}
03 DBScan Clustring -> model = DBSCAN():{
from sklearn.cluster import DBSCAN
x = df[['x','y']]
plt.scatter(x['x'],x['y'],cmap='rainbow')
##++++++++++++++
## Scalling 
from sklearn.preprocessing import StandardScaler
x_scaled = StandardScaler().fit_transform(x)
## Visualization
model = DBSCAN(min_samples=12 , eps=.4)
y_lables = model.fit_predict(x) ##(x_scaled)
plt.scatter(x['x'],x['y'],c=y_lables,cmap='rainbow')

import numpy as np                             
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_breast_cancer  
cancer = load_breast_cancer()
print(cancer.DESCR)
X = cancer.data ; y = cancer.target  ## features : X, y 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)## Split the data into training and testing sets

model = KNeighborsClassifier(n_neighbors=5) ## Initialize the model with 5 neighbors
kf = KFold(n_splits=5, shuffle=True, random_state=42) ## Define k-fold cross-validation
cross_val_score(model, X_train, y_train, cv=kf)  ## Perform cross-validation on the training set and get cross-validation scores and 5-fold cross-validation
y_train_pred = cross_val_predict(model, X_train, y_train, cv=kf) ## Use cross_val_predict to make predictions during cross-validation
accuracy_score(y_train, y_train_pred) ## evaluate the cross-validated predictions on the training set
}



def 21 02 Project for Hyperparameters Optimization:
    from sklearn.model_selection import GridSearchCV 
    ## knn params 
    param_grid = {'n_neighbors':[1,3,5,7,9]}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid , cv=5)
    grid_search.fit(X_train , y_train)
    grid_search.score(X_test,y_test)
    pd.DataFrame(grid_search.cv_results_)

def 22 ML_Project_bei_Kaggle:
    prj01:##dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
     pass

save_and_load_finalmodel_pickle:
    import pickle     
    model_name = 'final_model.sav'  ## save
    save_model = pickle.dump(rf3_model,open(model_name,'wb'))
    ##++++++++++++++++++++++++++++++++++++++++++ 
     my_model = pickle.load(open('final_model.sav','rb'))     ## load 
    my_model.predict(X_test)



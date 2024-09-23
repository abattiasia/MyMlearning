

# 016_df_PCA.py
    
# Code:
Code Ex.01 PCA on Iris dataset:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
##++++++++

## Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
##++++++++

## Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
##++++++++

## Apply PCA with 2 components
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
my_pca = pca.fit_transform(X_scaled)
print('ok')
my_pca  ##[:, 1]
##++++++++

## Plot the reduced data
plt.figure(figsize=(8, 6))
plt.scatter(my_pca[:, 0], my_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Species')
plt.show()
##+++++++++++++++++++++++++++++++++++++++++++++++++++

#Code Ex.02 PCA on Titanic dataset:
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
##++++++++

## Load the Titanic dataset (you can replace the path with your actual file location)
titanic_df = pd.read_csv('path/to/titanic.csv')
##++++++++

## Basic data cleaning (you can add more steps as needed):
## 1. Remove unnecessary columns (e.g., PassengerId, Name, Ticket, Cabin)
titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
##++++++++

## 2. Handle missing values (e.g., fill missing age values with the median)
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
##++++++++

## 3. Convert categorical variables to numerical (e.g., Sex, Embarked)
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['Embarked'] = titanic_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)
##++++++++

## Standardize features (mean = 0, variance = 1) for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(titanic_df.drop('Survived', axis=1))
##++++++++

## Apply PCA with desired number of components (e.g., 2)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
##++++++++

## Now X_pca contains the transformed features for PCA
## You can use X_pca for further analysis or modeling
## Print explained variance ratio (optional)
print("Explained variance ratio:", pca.explained_variance_ratio_)



#Explain:
print(f""" 01
Certainly! Principal Component Analysis (PCA) 
can indeed be used in classification models. Let’s explore how PCA fits into the context of classification:

PCA is a popular unsupervised learning technique used for dimensionality reduction. 
It helps simplify high-dimensional data by finding a new set of features (principal components) 
that capture most of the variance in the original data.

What is PCA?
PCA is a statistical technique used for dimensionality reduction.
It transforms high-dimensional data into a lower-dimensional space while retaining as much relevant information as possible.
The goal is to find a new set of features (principal components) that capture most of the variance in the original data.

Advantages of PCA in Classification:
Reduced Dimensionality: By selecting a smaller set of principal components, we reduce the number of features. 
This can significantly speed up training time and improve model performance.
Visualization: PCA allows us to visualize data in lower dimensions (e.g., 2D or 3D scatter plots) 
while preserving essential information.
Noise Reduction: It can remove noise or irrelevant features.
Applying PCA in Classification Models:

Data Preparation:
Normalize or standardize your features before applying PCA.
Ensure that categorical features are converted to numerical representations.

PCA Implementation:
Apply PCA to your feature matrix (X) to obtain transformed features (X_pca).
Choose the number of principal components based on the explained variance (e.g., retain components that 
explain a significant portion of the total variance).

Model Training:
Use the transformed features (X_pca) as input for your classification model (e.g., logistic regression, decision tree, or random forest).
Evaluate model performance using appropriate metrics (accuracy, precision, recall, etc.).

""")
print(f""" 02
https://www.simplilearn.com/tutorials/machine-learning-tutorial/principal-component-analysis
What is Principal Component Analysis (PCA)?
The Principal Component Analysis is a popular unsupervised learning technique for reducing the dimensionality of large data sets. 
It increases interpretability yet, at the same time, it minimizes information loss. It helps to find the most significant features in a dataset 
and makes the data easy for plotting in 2D and 3D. PCA helps in finding a sequence of linear combinations of variables.
PrincipalComponents
In the above figure, we have several points plotted on a 2-D plane. There are two principal components. PC1 is the primary principal 
component that explains the maximum variance in the data. PC2 is another principal component that is orthogonal to PC1.
Your AI/ML Career is Just Around The Corner!
AI Engineer Master's ProgramExplore ProgramYour AI/ML Career is Just Around The Corner!
What is a Principal Component?
The Principal Components are a straight line that captures most of the variance of the data. They have a direction and magnitude. 
Principal components are orthogonal projections (perpendicular) of data onto lower-dimensional space.
Now that you have understood the basics of PCA, let’s look at the next topic on PCA in Machine Learning. 
Dimensionality
The term "dimensionality" describes the quantity of features or variables used in the research. It can be difficult to visualize 
and interpret the relationships between variables when dealing with high-dimensional data, such as datasets with numerous variables. 
While reducing the number of variables in the dataset, dimensionality reduction methods like PCA are used to preserve the most crucial data. 
The original variables are converted into a new set of variables called principal components, which are linear combinations of the original 
variables, by PCA in order to accomplish this. The dataset's reduced dimensionality depends on how many principal components are 
used in the study. The objective of PCA is to select fewer principal components that account for the data's most important variation. 
PCA can help to streamline data analysis, enhance visualization, and make it simpler to spot trends and relationships between factors by 
reducing the dimensionality of the dataset.
The mathematical representation of dimensionality reduction in the context of PCA is as follows:
Given a dataset with n observations and p variables represented by the n x p data matrix X, the goal of PCA is to transform the original variables 
into a new set of k variables called principal components that capture the most significant variation in the data. The principal components are 
defined as linear combinations of the original variables given by:
PC_1 = a_11 * x_1 + a_12 * x_2 + ... + a_1p * x_p
PC_2 = a_21 * x_1 + a_22 * x_2 + ... + a_2p * x_p
...
PC_k = a_k1 * x_1 + a_k2 * x_2 + ... + a_kp * x_p
where a_ij is the loading or weight of variable x_j on principal component PC_i, and x_j is the jth variable in the data matrix X. The principal 
components are ordered such that the first component PC_1 captures the most significant variation in the data, the second component PC_2 captures 
the second most significant variation, and so on. The number of principal components used in the analysis, k, determines the reduced dimensionality of the dataset.
Correlation
A statistical measure known as correlation expresses the direction and strength of the linear connection between two variables. The covariance matrix, 
a square matrix that displays the pairwise correlations between all pairs of variables in the dataset, is calculated in the setting of PCA using correlation. The covariance matrix's diagonal elements stand for each variable's variance, while the off-diagonal elements indicate the covariances between different pairs of variables. The strength and direction of the linear connection between two variables can be determined using the correlation coefficient, a standardized measure of correlation with a range of -1 to 1.
A correlation coefficient of 0 denotes no linear connection between the two variables, while correlation coefficients of 1 and -1 denote 
the perfect positive and negative correlations, respectively. The principal components in PCA are linear combinations of the initial variables 
that maximize the variance explained by the data. Principal components are calculated using the correlation matrix.
In the framework of PCA, correlation is mathematically represented as follows:
The correlation matrix C is a nxn symmetric matrix with the following components given a dataset with n variables (x1, x2,..., xn):
Cij = (sd(xi) * sd(xj)) / cov(xi, xj)
where sd(x i) is the standard deviation of variable x i and sd(x j) is the standard deviation of variable x j, and cov(x i, x j) is the correlation 
between variables x i and x j.
The correlation matrix C can also be written as follows in matrix notation:
C = X^T X / (n-1) (n-1)
Orthogonal
The term "orthogonality" alludes to the principal components' construction as being orthogonal to one another in the context of the PCA algorithm. 
This indicates that there is no redundant information among the main components and that they are not correlated with one another.
Orthogonality in PCA is mathematically expressed as follows: each principal component is built to maximize the variance explained by it while adhering 
to the requirement that it be orthogonal to all other principal components. The principal components are computed as linear combinations of the original 
variables. Thus, each principal component is guaranteed to capture a unique and non-redundant part of the variation in the data.
The orthogonality constraint is expressed as:
a_i1 * a_j1 + a_i2 * a_j2 + ... + a_ip * a_jp = 0
for all i and j such that i ≠ j. This means that the dot product between any two loading vectors for different principal components is zero, indicating 
that the principal components are orthogonal to each other.
Master Gen AI Strategies for Businesses with
Generative AI for Business Transformation ProgramExplore ProgramMaster Gen AI Strategies for Businesses with
Eigen Vectors
The main components of the data are calculated using the eigenvectors. The ways in which the data vary most are represented by the eigenvectors of 
the data's covariance matrix. The new coordinate system in which the data is represented is then defined using these coordinates.
The covariance matrix C in mathematics is represented by the letters v 1, v 2,..., v p, and the associated eigenvalues are represented by _1, _2,..., _p. 
The eigenvectors are calculated in such a way that the equation shown below holds:
C v_i = λ_i v_i
This means that the eigenvector v_i produces the associated eigenvalue  λ_i as a scalar multiple of itself when multiplied by the covariance matrix C.
Covariance Matrix
The covariance matrix is crucial to the PCA algorithm's computation of the data's main components. The pairwise covariances between the factors in 
the data are measured by the covariance matrix, which is a p x p matrix.
The correlation matrix C is defined as follows given a data matrix X of n observations of p variables:
C = (1/n) * X^T X
where X^T represents X's transposition. The covariances between the variables are represented by the off-diagonal elements of C, whereas 
the variances of the variables are represented by the diagonal elements of C.
Steps for PCA Algorithm
Standardize the data: PCA requires standardized data, so the first step is to standardize the data to ensure that all variables have a mean of 0 and
 a standard deviation of 1.
Calculate the covariance matrix: The next step is to calculate the covariance matrix of the standardized data. This matrix shows how each variable is 
related to every other variable in the dataset.
Calculate the eigenvectors and eigenvalues: The eigenvectors and eigenvalues of the covariance matrix are then calculated. The eigenvectors represent 
the directions in which the data varies the most, while the eigenvalues represent the amount of variation along each eigenvector.
Choose the principal components: The principal components are the eigenvectors with the highest eigenvalues. These components represent 
the directions in which the data varies the most and are used to transform the original data into a lower-dimensional space.
Transform the data: The final step is to transform the original data into the lower-dimensional space defined by the principal components.
Applications of PCA in Machine Learning
PCA_Applications
PCA is used to visualize multidimensional data.
It is used to reduce the number of dimensions in healthcare data.
PCA can help resize an image.
It can be used in finance to analyze stock data and forecast returns.
PCA helps to find patterns in the high-dimensional datasets.
1. Normalize the Data
Standardize the data before performing PCA. This will ensure that each feature has a mean = 0 and variance = 1.
Zscore
2. Build the Covariance Matrix
Construct a square matrix to express the correlation between two or more features in a multidimensional dataset.
Covariance
3. Find the Eigenvectors and Eigenvalues
Calculate the eigenvectors/unit vectors and eigenvalues. Eigenvalues are scalars by which we multiply the eigenvector of the covariance matrix.
EigenValues%26Vectors
4. Sort the Eigenvectors in Highest to Lowest Order and Select the Number of Principal Components.
Now that you have understood How PCA in Machine Learning works, let’s perform a hands-on demo on PCA with Python.


""") 


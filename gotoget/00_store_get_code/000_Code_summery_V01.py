## get_mlclassification.py ##written by Dr. Attia 11.092024

get_code_import:{
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing  ## New dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
}

df_loaddataset:{
df = pd.read_excel('/kaggle/input/glass-imbalanced/glass (Imbalanced).xlsx')
df = pd.read_csv(/kaggle/input/glass-imbalanced/glass (Imbalanced).csv)
}

df_info:{
df.head(); df.info(); 
df.describe(); df.describe().T; 
df.shape; df.sample(); 
df.columns(); df.describe; 
df.columns;
df.isnull().count()
#+++
col='Location'
df[col].value_counts()  # to see the repeated value cont
#for all cols print counts in extra DataFrame:
value_counts = []    ## to calculate no of repeated values in each column
for col in df.columns:
    d = df[col].value_counts().count()
    value_counts.append(int(d))
pd.DataFrame(value_counts,index=df.columns).T   ## integriert value_counts with df.columns in df
#++++
#for all cols print each cat name and counts:
for col in df.select_dtypes(include=['category', 'object']).columns:
    print(f"Column: {col}")
    print(df[col].value_counts())
    print("\n")
}

df_cleaning:{
df.isnull().sum(); 
df.dropna(inplace=True)
null_cols = df.isnull().columns
df['TotalCharges'].isnull().sum()
df.dropna(subset=['col01'], inplace=True)
df.dropna(subset=df.isnull().columns, inplace=True)
df.dropna(inplace=True)

df1= df.fillna(df.mean(), inplace=True) ## only bei num_cols
df1= df['col_name'].fillna(df['col_name'].mean(), inplace=True) ## only bei num_cols
df= df['col_name'].fillna(df['col_name'].mode()[0], inplace=True) ##only bei cat_cols

df.duplicated().sum();  ## check and delete duplicate
df.drop_duplicates() 

df.drop(columns=['company'], inplace=True)  ## drop cols
df.drop(columns=['Evaporation'], inplace=True)  #Sunshine
df.drop(columns=['Sunshine'], inplace=True) 
df.drop(columns=['Date'], inplace=True) 
}

df_outlier:{
	
# user-defined function to apply z-score on data
# Built-in function ( language function ( Median(), abs() , Describe() , info() , Head() ))
import pandas as pd
import numpy as np

df = pd.Series( [78 , 85 , 92 , 70 , 65 , 88 , 95 , 60 , 150 ])
df
df.describe().T
abs_Median = abs ( df - df.median() )
abs_Median

mad = abs_Median.median()  # Build in Function
mad

# MZ ----> 0.6745 * (x - Median ) / Mad
df = 0.6745 * ( ( df - df.median() ) / mad )
df
# Determine a threshold ( Most commonly used is 3 )
df [ ( df < -3 ) | ( df > 3) ]

# Removing Outliers
df_no_Outlier = df [ ( df > -3 ) & ( df < 3) ]
df_no_Outlier



def ZScore_Outlier(df):
	abs_Median = abs ( df - df.median() )
	mad = abs_Median.median()
	print("MAD : ", mad)

	df = 0.6745 * ( ( df - df.median() ) / mad ) # MZ
	outlier = df [ ( df < -3 ) | ( df > 3) ]
	print(" Outliers : ", outlier)

	df_no_Outlier = df [ ( df > -3 ) & ( df < 3) ]

	return df_no_Outlier

df = pd.Series( [78 , 85 , 92 , 70 , 65 , 88 , 95 , 60 , 150 ])
df	
ZScore_Outlier(df)
df = pd.read_csv("/content/Example1.csv")
df
ZScore_Outlier(df['Income'])
#++++++++++++++++++++++++++++++++++
# Min Max Scaller
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

numpy_array = np.array ( [-400,-200,0,200,300,400,500,600,1000,1200])
scaler = MinMaxScaler() # Saclling between Zeer and One By defult
#reshape (-1,1 ) for the numpy array only , adding a data frame we will use df only
scaled = scaler.fit_transform(numpy_array.reshape(-1,1))
print(scaled)

df = pd.read_csv("/content/Example1.csv")
df
scaled = scaler.fit_transform(df['Income'].array.reshape(-1, 1))
scaled
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
scaled_df

}

df_handling_num_obj_cat_cols:{
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') ## change data type of cols
##+++++++++
#get all cols types
null_cols = df.isnull().columns
num_cols = df.select_dtypes(include=['number']).columns
obj_cols = df.select_dtypes(include=['object']).columns
cat_cols = df.select_dtypes(include=['category]).columns
cat_obj_cols = df.select_dtypes(include=['category', 'object']).columns
##++++
grouped_df = df.groupby('Category')['Value'].sum() ## was macht ? 
print(grouped_df)
##++++
cat_counts = df['HbA1c_level'].value_counts() ## was macht?
print(cat_counts)
##++++++++++++++
get_repeated_values_cat_or_num_cols:{
value_counts = []    ## to calculate no of repeated values in each column
for col in df.columns:
    d = df[col].value_counts().count()
    value_counts.append(int(d))
pd.DataFrame(value_counts,index=df.columns).T   ## help us to define which column cat oder stream
##+++++++
df['Mg'].unique()  ## kommt raus the values to define encoding 0, 1, 2
##++++++++++++++++++++++
num_cols = ['age','trtbps','chol','thalachh','oldpeak']
cat_cols = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
# make df from array 
x
pd.DataFrame(x,index=df.columns).T   ## integriert value_counts with df.columns in df
}
}



load.sns.iris:{  
# import
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('tips')
df.head()
plt.scatter(df.total_bill,df.tip)
#++
# visualize scatter : set context
plt.scatter(df.total_bill,df.tip)
sns.set_style('darkgrid')
sns.set_context('talk')
plt.show() # update chart
#+++
sns.color_palette()
#+++++
sns.relplot(data=df,x='total_bill',y='tip')
sns.relplot(data=df,x='total_bill',y='tip',hue="day")
sns.relplot(data=df,x='total_bill',y='tip',hue="sex")
sns.relplot(data=df,x='total_bill',y='tip',hue="sex",col='day')
sns.relplot(data=df,x='total_bill',y='tip',hue="sex",col='day',col_wrap=2,palette="Paired")
#+++
sns.pairplot(iris,palette='husl',hue='species')
plt.savefig('output2.png')

}

load.sns.titanic:{
titanic = sns.load_dataset('titanic')
titanic.head()
### Barplot
sns.barplot(x='embark_town',y='age',data=titanic)
sns.barplot(x='embark_town',y='age',data=titanic,palette='dark')
sns.barplot(x='embark_town',y='age',data=titanic,palette='dark',hue='sex')
sns.barplot(x='embark_town',y='age',data=titanic,palette='dark',hue='sex',errorbar=None)
sns.barplot(y='embark_town',x='age',data=titanic,palette='dark',hue='sex',errorbar=None,orient='h')

#++++
sns.histplot(titanic['age'])
sns.histplot(titanic['age'],kde=True)
#++++
sns.countplot(x='class',data=titanic)
	
	
	
}

array_Visualizatin::{
data = [1,2,3,3,3,3,4,4,5,5,5,5,6,6]
plt.hist(data) # or (data,bins=5)
# with outliers
data = [1,8,10,8,8,10,20,100]
plt.boxplot(data)
# without outliers
data = [4,8,10,8,8,10,15,18,20]
plt.boxplot(data)
#++++++++
sns.countplot(x='class',data=titanic)
sns.countplot(x='class',data=titanic,palette='magma',hue='sex')
#++  .jointplot
sns.jointplot(x='total_bill',y='tip',data=df)
sns.jointplot(x='total_bill',y='tip',data=df,kind='reg')
sns.jointplot(x='total_bill',y='tip',data=df,kind='kde')
sns.jointplot(x='total_bill',y='tip',data=df,kind='hex')
plt.savefig('output3.png')
#++
sns.boxplot(x='day',y='total_bill',data=df)
#+++
sns.pairplot(iris,palette='husl',hue='species')
plt.savefig('output2.png')





	
}

df_Visualizatin:{

plot.line:  ##simple plot
df['House_Price'].head(100).plot();   
#+++

#+++++
### Barplot
sns.barplot(x='embark_town',y='age',data=titanic)
sns.barplot(y='embark_town',x='age',data=titanic,palette='dark',hue='sex',errorbar=None,orient='h')
#++++ scatterplot
sns.scatterplot(x='Square_Footage', y='House_Price', data=df) 
#++ pairplot
sns.pairplot(df.head(10))
#++ histplot  -> only num_cols
sns.histplot(titanic['age'])
sns.histplot(titanic['age'],kde=True)
#++ countplot
sns.countplot(x='class',data=titanic)
sns.countplot(x='class',data=titanic,palette='magma',hue='sex')
#++  .jointplot
sns.jointplot(x='total_bill',y='tip',data=df)
sns.jointplot(x='total_bill',y='tip',data=df,kind='reg')
sns.jointplot(x='total_bill',y='tip',data=df,kind='kde')
sns.jointplot(x='total_bill',y='tip',data=df,kind='hex')
plt.savefig('output3.png')
#++++
sns.boxplot(x='day',y='total_bill',data=df)

#++relplot
sns.relplot(data=df,x='total_bill',y='tip')
sns.relplot(data=df,x='total_bill',y='tip',hue="day")
sns.relplot(data=df,x='total_bill',y='tip',hue="sex")
sns.relplot(data=df,x='total_bill',y='tip',hue="sex",col='day')
sns.relplot(data=df,x='total_bill',y='tip',hue="sex",col='day',col_wrap=2,palette="Paired")

#+++++
df['TotalCharges'].head(100).plot.line();      
plt.pie _col01.value_counts()
#+++ save fig:  add this line of code with chart in the same cell
plt.savefig('output.png')



}


df_matplotlib_plt:{

# visualize scatter : matplotlib
df.head()
plt.scatter(df.total_bill,df.tip)
#+++
# visualize scatter : set context
plt.scatter(df.total_bill,df.tip)
sns.set_style('darkgrid')
sns.set_context('talk')
plt.show() # update chart


##+++++++++++++++++           
plt.scatter:{
x = df[['x','y']]; 
y_labels=df['color'] ## x,y and color are cols in df
plt.scatter(x['x'],x['y'],c=y_labels,cmap='rainbow')
##++++++++++++++++
x = df[['x','y']]
plt.scatter(x['x'],x['y'],cmap='rainbow') ## simple without point color
##+++++++++++++++++++++++++
plt.figure(figsize=(10,6))
plt.scatter(df['class'], df['Flavanoids'], c=None, edgecolor='m', alpha=0.75)##c is only point color
plt.grid(True)
plt.title('Scatter plot 2 Variables- \n in df', fontsize=15)
plt.xlabel("wine class")
plt.ylabel("Alcohol")
plt.show() }

plt_hist(variable):{   ##plot for all num_cols in df
plt.figure(figsize= (9,3))
plt.hist(dftrain[variable], bins=100)
plt.xlabel(variable)
plt.ylabel("frequency")
plt.title("{} distrubition with hist".format(variable))
plt.show()
num_cols =["Fare", "Age", "PassengerId"]
num_cols = df.select_dtypes(include=['number']).columns
for n in num_cols:
	plt_hist(n)}
	
plt.pie:{
gender_counts = df['gender'].value_counts()
## رسم المخطط الدائري
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['ff9999','66b3ff'])
plt.title('Distribution of Gender')
plt.axis('equal')  ## Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
plt.pie_divided_col01_to_cats:
## تقسيم العملاء إلى فئات حسب مدة الاشتراك (tenure)
bins = [0, 12, 24, 48, 60, 72]
labels = ['0-12 months', '12-24 months', '24-48 months', '48-60 months', '60-72 months']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
## حساب توزيع العملاء حسب الفئات
tenure_group_counts = df['tenure_group'].value_counts().sort_index()
## رسم المخطط الدائري
plt.figure( figsize =(8, 6))
plt.pie( tenure_group_counts, labels=tenure_group_counts.index, autopct='%1.1f%%', startangle=140, colors=['ff9999', '66b3ff', '99ff99', 'ffcc99', 'c2c2f0'])
plt.title(' Distribution of Customers by Tenure Group')
plt.axis('equal')  ## Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
plt.pie_2_cols_in_subplot:
## حساب توزيع العملاء حسب خدمة الهاتف (PhoneService)
phone_service_counts = df['PhoneService'].value_counts()
## حساب توزيع العملاء حسب الأمان الإلكتروني (OnlineSecurity)
online_security_counts = df['OnlineSecurity'].value_counts()
## رسم المخطط الدائري لخدمة الهاتف
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.pie(phone_service_counts, labels=phone_service_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff'])
plt.title('Distribution of Customers by Phone Service')
plt.axis('equal')  ## Equal aspect ratio ensures that pie is drawn as a circle.
## رسم المخطط الدائري للأمان الإلكتروني
plt.subplot(1, 2, 2)
plt.pie(online_security_counts, labels=online_security_counts.index, autopct='%1.1f%%', startangle=140, colors=['##99ff99', '##ffcc99', '##c2c2f0'])
plt.title('Distribution of Customers by Online Security')
plt.axis('equal')
plt.tight_layout()
plt.show()     
plt.pie_3_cols_in_subplot:
## رسم المخطط الدائري لعمود StreamingTV
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
streaming_tv_counts = df['StreamingTV'].value_counts()
plt.pie(streaming_tv_counts, labels=streaming_tv_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff', '##99ff99'])
plt.title('Distribution of Customers by Streaming TV')
## رسم المخطط الدائري لعمود StreamingMovies
plt.subplot(1, 3, 2)
streaming_movies_counts = df['StreamingMovies'].value_counts()
plt.pie(streaming_movies_counts, labels=streaming_movies_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff', '##99ff99'])
plt.title('Distribution of Customers by Streaming Movies')
## رسم المخطط الدائري لعمود Contract
plt.subplot(1, 3, 3)
contract_counts = df['Contract'].value_counts()
plt.pie(contract_counts, labels=contract_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff', '##99ff99'])
plt.title('Distribution of Customers by Contract')
plt.show() }
 }    
 
df_seaborn_sns:{
sns.boxplot_features:{
	features = df.iloc[:,:9].columns.tolist()
	plt.figure(figsize=(18, 27))
	for i, col in enumerate(features):
		plt.subplot(6, 4, i*2+1)
		plt.subplots_adjust(hspace =.25, wspace=.3)
sns.boxplot(y = col, data = df, x="target", palette = ["blue", "yellow"]) }
##+++++++++++++++++
sns.countplot:{
def sns.countplot:                ## plot col vs target
	## sns.countplot
	plt.figure(figsize=(10, 5))
	col02=' '
	target = ''
	sns.countplot(x=col01, hue=target, data=df, palette=['darkturquoise', 'royalblue'])
	plt.title('Count of target by col01')
	plt.xlabel('col01')
	plt.ylabel('Count')
	plt.legend(title='target', labels=['No target', 'target])
	plt.show()
	return
cat_cols = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']
for c in cat_cols:
	sns.countplot(x=c,data=df,palette='Set2',hue='output')
	plt.title(f'Count {c}')
	plt.show()
}
##++++++++++++++++++
sns.histplot_col01_with_hue:{
## ضبط حجم الرسم البياني
plt.figure(figsize=(12, 8))
## استخدام Seaborn لإنشاء مخطط مكدس لتوزيع العقود بناءً على "StreamingTV"
plt.subplot(2, 1, 1)
sns.histplot(data=df, x="Contract", hue="StreamingTV", multiple="stack", palette="Set2", edgecolor=".3")
plt.title('Contract Distribution by StreamingTV', fontsize=16)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
## استخدام Seaborn لإنشاء مخطط مكدس لتوزيع العقود بناءً على "StreamingMovies"
plt.subplot(2, 1, 2)
sns.histplot(data=df, x="Contract", hue="StreamingMovies", multiple="stack", palette="Set1", edgecolor=".3")
plt.title('Contract Distribution by StreamingMovies', fontsize=16)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.show()
plt.pie_df.groupby:
## توزيع العقود بناءً على فئة StreamingTV
streaming_tv_contract_counts = df.groupby('StreamingTV')['Contract'].value_counts().unstack()
plt.figure(figsize=(10, 6))
for i, tv_option in enumerate(streaming_tv_contract_counts.index):
	plt.subplot(1, len(streaming_tv_contract_counts.index), i + 1)
	plt.pie(streaming_tv_contract_counts.loc[tv_option], labels=streaming_tv_contract_counts.columns, autopct='%1.1f%%', startangle=140)
	plt.title(f'Contracts for {tv_option} StreamingTV')
plt.tight_layout()
plt.show()
## توزيع العقود بناءً على فئة StreamingMovies
streaming_movies_contract_counts = df.groupby('StreamingMovies')['Contract'].value_counts().unstack()
plt.figure(figsize=(10, 6))
for i, movies_option in enumerate(streaming_movies_contract_counts.index):
	plt.subplot(1, len(streaming_movies_contract_counts.index), i + 1)
	plt.pie(streaming_movies_contract_counts.loc[movies_option], labels=streaming_movies_contract_counts.columns, autopct='%1.1f%%', startangle=140)
	plt.title(f'Contracts for {movies_option} StreamingMovies')
plt.tight_layout()
plt.show()      
sns.countplot_to_define_relation_between_2Cols:
plt.figure(figsize=(10, 5))
sns.countplot(x='heart_disease', hue='diabetes', data=df, palette=['darkturquoise', 'royalblue'])
plt.title('Count of Diabetes')
plt.xlabel('heart_disease')
plt.ylabel('Count')
plt.legend(title='diabetes', labels=['Not diabetes', 'diabetes'])
plt.show()
}

}

df_plotly_px:{
#plot for 2 cols. in df
fig = px.bar(
	df, x='name', y='price',  color='category', 
	title='Expected Profit by Subject and Level',
	labels={'subject': 'Subject', 'expected_profit': 'Expected Profit (USD)'},
	hover_data=['price', 'price', 'price']  ## Add more details on hover
    title_font_size=20,  ## Increase title font size
	xaxis_title_font_size=14,  ## Increase x-axis label font size
	yaxis_title_font_size=14,  ## Increase y-axis label font size
	hoverlabel=dict( bgcolor="white", font_size=12, font_family="Rockwell" )
)
fig.show()
}  

df_Encoding_obj_cols:{
/*-> By_LabelEncoder:*/
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])   ## 01 Encoding: labelEncoding

/*-> By_get_dummies:*/
df = pd.get_dummies(data=df,columns=['Gender'],drop_first=True)  ## 02 Encoding by dummies values

/*##+++++++++*/
categorical_columns = data.select_dtypes(include=['category', 'object']).columns
X_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
X_encoded.describe().T
##++++++++++++++++
By_one-hot:
pass
By_Binary encoding:
pass
}   

df_correlation_matrix:{
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cbar = True, annot =True,cmap='twilight_shifted_r')
plt.title('correlation matrix',fontdict={'fontsize': 20},fontweight ='bold')
plt.savefig("corr_matrix.png")
}

df_scaling_StandardScaler_or_ Normlizing:{
/* Info_Feature_Scaling(f""" Feature Scaling
- When your data has different values, and even different measurement units, it can be difficult to compare them. 
	What is kilograms compared to meters? Or altitude compared to time?
- if the values of the features are closer to each other
	 there are chances for the algorithm to get trained well and faster instead of the data set where the data points or 
	 features values have high differences with each other will take more time to understand the data and the accuracy will be
lower.
- We can scale data into new values that are easier to compare.
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGh1NUJds4XLsZt22F9Mg0yD2OVeQGVs2tyg&s)

Example
![](https://ashutoshtripathi.com/wp-content/uploads/2021/06/image-3.png?w=600&h=280&crop=1)
- Feature scaling is essential for machine learning algorithms that calculate distances between data:
	- K-nearest neighbors
	- Regression
	- kmeans
	- Principal Component Analysis(PCA)
- If not scale, the feature with a higher value range starts dominating when calculating distances

How to perform
- Scaling types :
	1) Min Max Scaler
	2) Standard Scaler
	3) Max Abs Scaler
	4) Robust Scaler
	5) Quantile Transformer Scaler
	6) Power Transformer Scaler
	7) Unit Vector Scaler
![](https://ourcodingclub.github.io/assets/img/tutorials/data-scaling/penguin_panel.png)

	Min-Max  ->  x' = (x-x(min))/(x(max)-x(min))
	- Transform features by scaling each feature to a given range.
	- scales and translates each feature individually
	- This Scaler shrinks the data within the range of -1 to 1 if there are negative values.
	- We can set the range like [0,1] or [0,5] or [-1,1].
	- This Scaler responds well if the standard deviation is small and when a distribution is not Gaussian.
	- This Scaler is sensitive to outliers.
	![](https://www.atoti.io/wp-content/uploads/2024/06/1_lz4NqpzsmNH9bvcZttvcCg.webp)

	""")  */
Scaling_StandardScaler=Z-Score-> Standardization -> for all x in df without y:{
Explain:/*
print(f"""
Purpose: StandardScaler aims to transform your data so that 
it follows a standard normal distribution (SND). 
In other words, it makes the mean of each feature equal to 0 
and scales the data to have a unit variance.
Scaling Process: It subtracts the mean from each feature value X.
Then, it divides by the standard deviation.
Mathematically: For a feature (X), the transformed value (X_{\text{new}}) 
is calculated as: [ X_{\text{new}} = \frac{{X - \text{mean}}}{{\text{Std}}} ]
Formula: (z = \frac{{x - \mu}}{{\sigma}}), where (x) is the original value, (\mu) is the mean, and (\sigma) is the standard deviation.
Widely used and robust against outliers.

Range: The scaled features can take any real value, and 
they typically have a mean of 0 and a standard deviation of 1.

Use Case: StandardScaler is useful when you want to ensure that your 
features have similar scales and are centered around zero. It’s commonly 
used in algorithms that assume normally distributed data, 
such as linear regression or logistic regression.
""") */
code:
from sklearn.preprocessing import StandardScaler
sacler = StandardScaler(); 
x = sacler.fit_transform(x)    ## 01 Same Model With Scaling 
x # now x is array 
#you can make x as df
x_as_df = pd.DataFrame(x, columns=df.columns)
}
Scaling_MinMaxScaler -> Normlization for all x in df without y :{
Explain:/*
print(f""" 01
MinMaxScaler:
Purpose: 
MinMaxScaler scales all the features in your dataset to a specific range, usually [0, 1]. 
If your data contains negative values, it can also scale them to the range [-1, 1].
Scaling Process: 
It subtracts the minimum value from each feature value X.
Then, it divides by the range (maximum value minus minimum value).
Range: The scaled features will always fall within the specified range.
Scales features to a range between 0 and 1.
Formula: (x_{\text{normalized}} = \frac{{x - x_{\text{min}}}}{{x_{\text{max}} - x_{\text{min}}}})
Useful when you want to preserve the original distribution.
Use Case: 
MinMaxScaler is commonly used when you want to normalize your features 
to a consistent range. It’s helpful for algorithms that rely on distances or gradients, 
such as k-means clustering or neural networks.
Note: 
MinMaxScaler is sensitive to outliers, so if your dataset has extreme values, 
consider using other scalers or preprocessing methods
""")*/
code:
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  
scaled = scaler.fit_transform(df.values)     
print(scaled)
scaled_df = pd.DataFrame(scaled, columns=df.columns)
print(scaled_df)
}
}

df_PCA:{
Explain:{
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
    
    
    """) }
Code Ex.01 PCA on Iris dataset:{
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
}
Code Ex.02 PCA on Titanic dataset:{
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
}
}
}

Ml_Regression:{
model = LinearRegression()
model.fit(X_train , y_train)
model.coef_
model.intercept_
y_pred = model.predict(X_test)
y_pred
### Model Evaluation
from sklearn.metrics import r2_score
r2_score(y_test , y_pred)

}

ML_Classification:{
00 define x,y and X_train , X_test for ML Model:{
target = 'diabetes'
x= df_encoded.drop(columns=[target])  
y = df_encoded[target]
##+++++++++
x = df.drop(columns=['Class'], inplace=False) 
y = df['Class']
## data split 
X_train , X_test , y_train , y_test = train_test_split(x,y,test_size=.20)
##++++++++++++
05 Classification report:
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
##++++++++
 06 Comparison  between the 3 Models:
##KNeighborsClassifier() accuracy =  92%
##DecisionTreeClassifier() accuracy = 81.5%
##RandomForestClassifier()  accuracy = 82.57%
##++++++++
 07 lazypredict to Compare Classification Models  lib -->  using lazypredict lib:
## [lazypredict](https://github.com/shankarpandala/lazypredict)
##https://github.com/shankarpandala/lazypredict
##!pip install lazypredict
from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
  print(models);    }   
01 KNN Prject  K Neighbors Classifier  -> model=KNeighborsClassifier():{
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
## build model 
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train , y_train)
print('ok')
## predict 
y_pred = model.predict(X_test)
y_pred
## accuracy 
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
  plt.show()}
02 Tree Prject  Decision  Tree  Classifier -> model=DecisionTreeClassifier():{
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
  plt.show()}
03 Randum Project Random Forest --> model=RandomForestClassifier():{
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
  }
04 Naive Bayes classifier  Build Model:{
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
  }
Apply_cross_Validation-> to get better accurcey:{
pass
  }
Apply_Gridsearch_to_define_best_model_params:{
Ex.01_apply_to_ RandomForestClassifier:
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
## Assuming you have already split your data into X_train, X_test, y_train, and y_test
## Replace 'X_train', 'X_test', 'y_train', and 'y_test' with your actual data
## Define the parameter grid
param_grid = {
	'n_estimators': [30],  ## Set the desired number of estimators
	'criterion': ['entropy']  ## Set the criterion to 'entropy'
}
## Initialize GridSearchCV with RandomForestClassifier
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
## Fit the model to the training data
grid_search.fit(X_train, y_train)
## Evaluate the model on the test data
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
## Create a DataFrame with the results
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df)
##++++++++++++++++++++++++++++++++++
##Ex.02
param_grid = {
'n_estimators': [50, 100, 150],
'max_depth': [None, 10, 20, 30],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4],
'bootstrap': [True, False]}
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
						   param_grid=param_grid,
						   cv=5,  
						   n_jobs=-1)
grid_search.fit(X_train, y_train)
##+
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")
##+
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with Best Parameters: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
Ex.02_apply_to_KNeighborsClassifier:
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
## Assuming you have already split your data into X_train, X_test, y_train, and y_test
## Replace 'X_train', 'X_test', 'y_train', and 'y_test' with your actual data
## Define the parameter grid
param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15]}
## Initialize GridSearchCV with KNeighborsClassifier
grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
## Fit the model to the training data
grid_search.fit(X_train, y_train)
## Evaluate the model on the test data
test_score = grid_search.score(X_test, y_test)
print(f"Test score: {test_score:.4f}")
## Create a DataFrame with the results
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df)
}

ML_Clustering:{
00 prepear_dataset:{
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
}
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

save_and_load_finalmodel_pickle:{
    import pickle     
    model_name = 'final_model.sav'  ## save
    save_model = pickle.dump(rf3_model,open(model_name,'wb'))
    ##++++++++++++++++++++++++++++++++++++++++++ 
     my_model = pickle.load(open('final_model.sav','rb'))     ## load 
    my_model.predict(X_test)
}

ML_Projects by Kaggle.com:{
01_ML_project_classification_Hotel:{
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
}

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
}



def 02 ML_Project_Classification_13092024:{
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
params = {
	'n_estimators' : [10,25,50,100,150,200],
	'criterion' : ['entropy' , 'gini' , 'log_loss']}
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

}
   
def 200 DL Notebook am 13092024:{
Part 01:
get_MyLinks:
	print('https://playground.tensorflow.org")
Deep learning
	- Type of machine learning inspired by human brains
	- Structure : Artificial neural networks
	- It needs a massive number of data to be trained
	- It also needs high GPU
	- It takes time to be trained
	- It is the most used
	![](https://www.researchgate.net/profile/Umair-Shahzad/publication/357631533/figure/fig1/AS:1109568985808903@1641553269731/Machine-learning-as-a-subfield-of-artificial-intelligence_Q320.jpg)

Famous frameworks :
		- Tensorflow
		- Keras
		- Pytorch
		- Dl4j
		- Caffe
		- Microsoft cognitive toolkit
	![](https://miro.medium.com/v2/resize:fit:467/1*rJFONqrZEN7y7cwrVt3lXw.jpeg)

Deep learning vs Machine Learning
	- Deep learning will give you a lot of benefits :
		- No dimensionality reduction
		- No feature extraction
		- Deal with structured and un structured data (audio , images , videos , even text)
		- more data more accurate and performance
		- Complex problems
	![](https://images.prismic.io/turing/652ebc26fbd9a45bcec81819_Deep_Learning_vs_Machine_Learning_3033723be2.webp?auto=format%2Ccompress&fit=max&w=3840)

	- But with cost :
		- A lot of Training time
		- High Computational power GPU
		- Huge amount of data
		- More data will lead to saturation

Why not deep learning :
	- Simple problems
	- Acceptable error
	- Small data
	
DL Applications
	- Customer support
	- Medical care
	- Self driving cars
	- Face recognition
	- Object detection
	- Recommendations
	- Robotics

DL steps
	- The network will be layers on neurons
	- First layer : input layer
	- Last layer : output layer
	- In the middle : the hidden layers (for computations , feature extraction)
	- The input data will be a flatten matrix to the first layer (each pixel to a neuron)
	- Neurons for a layer are connected to the neurons of the next layer using channels
	- Each channel assigned a value = weight
	- The neuron value will be multiplied by the weight + the bias (neuron value)
	- The result value will be passed to a threshold value called = activation function
	- The result of the activation function will determine if the particular neurons will get activated or not
	- Activated neuron will pass the data to the neurons of the next layer using the channels
	- And so on the data will be propagated through the network (forward progradation)
	- In the output layer the neuron with highest value (probability) fires the output (predicted)
	- During this process the network will compare the predicted output with the real output to realize the error (backward propagation)
	- Based on the this the weights are being adjusted
	- This process will continue until the the network can predict the output correctly (most of the cases)
	![](https://miro.medium.com/v2/resize:fit:1400/1*OGFvJgMe21_5fCzUUyLwrw.png)

ANN
- Build on top of biological neuron network
- The simplest ann consist of 1 hidden layer
- Many layers = deep neural network
- Each neuron connects to another and has an associated weight and threshold
- The first weights are generated randomly , then be optimized
- If the output of any individual neuron is above the threshold value , that node is activated (send data to the next layer) if not no data will be passed (this feature is not important)
- There are many hidden layer types , most general is dense layer
- All deep learning networks build on top of gradient descent
![](https://miro.medium.com/v2/resize:fit:933/1*wBhuiErzkNMiKa1yklXbZA.png)

Activation function 
- defines how the weighted sum on input is transformed into an output
![](https://media.licdn.com/dms/image/D4D12AQH2F3GJ9wen_Q/article-cover_image-shrink_720_1280/0/1688885174323?e=2147483647&v=beta&t=gFWxErTLLWBc6iRWDxCBRxkdJ7ob24cmjWZAOuKN9o4)

Loss function :
- Find the difference between expected sand the predicted
![](https://miro.medium.com/v2/resize:fit:616/1*N1PyOYeog-vyytRbwEwQCQ.png)

Optimizer :
- Used to change the attributes of the neural network
(weight , batch size , learning rater ) to reduce the loss
- Determine how the network will be updated next epoch
- Weight updating process called backpropagation
![](https://miro.medium.com/v2/resize:fit:1200/1*Vi667n-YgG04HLVoi95ChQ.png)
![](file:///G:/My_Drive/MyPyWork/bilder/optimizer.png)  

Advantages :
- ANN has the ability to learn and model non-linear and complex relationships as many relationships between input and output are non-linear.
- After training, ANN can infer unseen relationships from unseen data, and hence it is generalized.
- Unlike many machine learning models, ANN does not have restrictions on datasets like data should be Gaussian distributed or nay other distribution.

Applications:
- Image Preprocessing and Character Recognition.
- Forecasting.
- Credit rating.
- Fraud Detection.

Layers :
Input Layer
	- Purpose: The input layer is where the neural network receives the data that will be processed.
	- Structure: Each neuron in this layer corresponds to a feature or attribute in the input data.
	- For example, if your dataset has 5 features (age, sex, cholesterol, etc.), the input layer will have 5 neurons.
	- How It Works: The values of the input data (e.g., pixel values in an image, or features in a tabular dataset) are fed into the network via the input layer. No learning
	happens here; it simply passes the data to the next layer.

Hidden Layers
	- Purpose: The hidden layers are where the actual learning happens in the neural network. They transform the input data into useful representations.
	- Structure: A deep learning model can have multiple hidden layers, and each hidden layer can have a different number of neurons. The number of neurons in each hidden layer is a hyperparameter that is usually determined through experimentation.
	- How It Works:
		- Each neuron in the hidden layers performs a weighted sum of the inputs it receives from the previous layer (input layer or another hidden layer).
		- The weighted sum is passed through an activation function (like ReLU, Sigmoid, etc.), introducing non-linearity, enabling the model to learn complex patterns.
		- The hidden layers progressively extract features from the data. Early layers may extract simple features, while deeper layers combine these into more complex patterns.
		- The process is repeated for each hidden layer, with the output from one layer feeding as input to the next.

Output Layer
	- Purpose: The output layer is where the final decision or prediction is made.
	- Structure: The number of neurons in the output layer depends on the task:
	- Classification tasks: For binary classification (e.g., "yes" or "no"), there will be 1 neuron in the output layer. For multi-class classification (e.g., categorizing something into 5 classes), there will be as many neurons as there are classes.
	- Regression tasks: There is typically one neuron in the output layer, representing the predicted continuous value.      
	- How It Works:
		- In a classification task, the neurons in the output layer use an activation function like softmax (for multi-class) or sigmoid (for binary classification) to produce probabilities or output scores.
		- In regression tasks, the activation function may be a linear function (or no activation at all), allowing the output to be a continuous value.
		 ![](https://miro.medium.com/v2/resize:fit:1199/1*N8UXaiUKWurFLdmEhEHiWg.jpeg)

Chalenging Questions:
- How many hidden layers should be selected ?
- How many neurons in each layer should be ?
- Which activation function should be selected ?
- In case of training, what is the number of epochs and batches ?
}

def DL_Code_in_Ex.NeuralNetwork:{
import numpy as np
def sigmoid(x):
  ## Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))
def deriv_sigmoid(x):
  ## Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)
def mse_loss(y_true, y_pred):
  ## y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()
class OurNeuralNetwork:
  print(f '''
  A neural network with:
	- 2 inputs
	- a hidden layer with 2 neurons (h1, h2)
	- an output layer with 1 neuron (o1)

*** DISCLAIMER ***:
The code below is intended to be simple and educational, NOT optimal.
Real neural net code looks nothing like this. DO NOT use this code.
Instead, read/run it to understand how this specific network works.
''')
def __init__(self):
	## Weights
	self.w1 = np.random.normal()
	self.w2 = np.random.normal()
	self.w3 = np.random.normal()
	self.w4 = np.random.normal()
	self.w5 = np.random.normal()
	self.w6 = np.random.normal()

	## Biases
	self.b1 = np.random.normal()
	self.b2 = np.random.normal()
	self.b3 = np.random.normal()
	
	def feedforward(self, x):
		## x is a numpy array with 2 elements.
		h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
		h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
		o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
		return o1

def train(self, data, all_y_trues):
	'''
	- data is a (n x 2) numpy array, n = ## of samples in the dataset.
	- all_y_trues is a numpy array with n elements.
	  Elements in all_y_trues correspond to those in data.
	'''
	learn_rate = 0.1
	epochs = 1000 ## number of times to loop through the entire dataset

	for epoch in range(epochs):
	  for x, y_true in zip(data, all_y_trues):
		## --- Do a feedforward (we'll need these values later)
		sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
		h1 = sigmoid(sum_h1)

		sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
		h2 = sigmoid(sum_h2)

		sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
		o1 = sigmoid(sum_o1)
		y_pred = o1

		## --- Calculate partial derivatives.
		## --- Naming: d_L_d_w1 represents "partial L / partial w1"
		d_L_d_ypred = -2 * (y_true - y_pred)

		## Neuron o1
		d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
		d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
		d_ypred_d_b3 = deriv_sigmoid(sum_o1)

		d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
		d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

		## Neuron h1
		d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
		d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
		d_h1_d_b1 = deriv_sigmoid(sum_h1)

		## Neuron h2
		d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
		d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
		d_h2_d_b2 = deriv_sigmoid(sum_h2)

		## --- Update weights and biases
		## Neuron h1
		self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
		self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
		self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

		## Neuron h2
		self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
		self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
		self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

		## Neuron o1
		self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
		self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
		self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

		## --- Calculate total loss at the end of each epoch
		if epoch % 10 == 0:
			y_preds = np.apply_along_axis(self.feedforward, 1, data)
			loss = mse_loss(all_y_trues, y_preds)
			print("Epoch %d loss: %.3f" % (epoch, loss))
	
## Define dataset
data = np.array([
  [-2, -1],  ## Alice
  [25, 6],   ## Bob
  [17, 4],   ## Charlie
  [-15, -6], ## Diana
])
all_y_trues = np.array([
  1, ## Alice
  0, ## Bob
  0, ## Charlie
  1, ## Diana
])
## Train our neural network!
network = OurNeuralNetwork()
network.train(data, all_y_trues)
using
pass
}

def 90 NLP:
   pass

new 


def von_Eng._Mahmoud_django???:
        from django.shortcuts import render , redirect
        from django.views.generic import ListView , DetailView
        from django.db.models import Q , F , Func , DecimalField , Value , CharField
        from django.db.models.aggregates import Avg , Sum , Count , Max , Min
        from .models import Product , Brand , Review , ProductImages
        from .forms import ReviewForm
        from django.db.models.functions import Cast
        from django.views.decorators.cache import cache_page
        from .tasks import send_emails
        from django.http import JsonResponse
        from django.template.loader import render_to_string
}
link:{
	
https://math.mit.edu/ennui
https://math.mit.edu/ennui/#%7B%22graph%22:%5B%7B%22children_ids%22:%5B2%5D,%22id%22:0,%22layer_name%22:%22Input%22,%22params%22:%7B%22dataset%22:%22mnist%22%7D,%22parent_ids%22:%5B%5D,%22xPosition%22:97.83081340789795,%22yPosition%22:305.08459091186523%7D,%7B%22children_ids%22:%5B3%5D,%22id%22:2,%22layer_name%22:%22Conv2D%22,%22params%22:%7B%22filters%22:16,%22kernelSize%22:%5B3,3%5D,%22strides%22:%5B1,1%5D,%22kernelRegularizer%22:%22none%22,%22regScale%22:0.1,%22activation%22:%22relu%22%7D,%22parent_ids%22:%5B0%5D,%22xPosition%22:250,%22yPosition%22:243.2%7D,%7B%22children_ids%22:%5B4%5D,%22id%22:3,%22layer_name%22:%22Flatten%22,%22params%22:%7B%7D,%22parent_ids%22:%5B2%5D,%22xPosition%22:571.4285714285714,%22yPosition%22:243.2%7D,%7B%22children_ids%22:%5B1%5D,%22id%22:4,%22layer_name%22:%22Dense%22,%22params%22:%7B%22units%22:32,%22activation%22:%22relu%22%7D,%22parent_ids%22:%5B3%5D,%22xPosition%22:769.2307692307693,%22yPosition%22:243.2%7D,%7B%22children_ids%22:%5B%5D,%22id%22:1,%22layer_name%22:%22Output%22,%22params%22:%7B%7D,%22parent_ids%22:%5B4%5D,%22xPosition%22:900,%22yPosition%22:304%7D%5D,%22hyperparameters%22:%7B%22batchSize%22:64,%22epochs%22:6,%22learningRate%22:0.01%7D%7D
	
https://medium.com/   -> all  account: 
	
}

Dataset_links:{
https://keras.io/api/datasets/cifar10/
	
}
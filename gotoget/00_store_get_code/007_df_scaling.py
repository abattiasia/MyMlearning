
# 007_df_scaling.py

# 01 code:StandardScaler();
from sklearn.preprocessing import StandardScaler
sacler = StandardScaler(); 
x = sacler.fit_transform(x)    ## 01 Same Model With Scaling 
x # now x is array

#  if you want make df from array
x_as_df = pd.DataFrame(x, columns=df.columns)
#++++++++++++++++++++++++++++++++++++

# 02 code: MinMaxScaler()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()  
scaled = scaler.fit_transform(df.values)     
print(scaled)

# if you want make df from array
scaled_df = pd.DataFrame(scaled, columns=df.columns)
print(scaled_df)
#++++++++++++++++++++++++++++++++++++

# Example to understand

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

#+++++++++++++++++++++++++++++++++++++++++++++++++

#df_scaling_StandardScaler_or_ Normlizing
print(f""" Feature Scaling
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

""")


#Scaling_StandardScaler=Z-Score-> Standardization -> for all x in df without y
#Explain:
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
""")
      
#code:
from sklearn.preprocessing import StandardScaler
sacler = StandardScaler(); 
x = sacler.fit_transform(x)    ## 01 Same Model With Scaling 
x # now x is array 
#you can make x as df
x_as_df = pd.DataFrame(x, columns=df.columns)


#Scaling_MinMaxScaler -> Normlization for all x in df without y :
#Explain:/*
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




# 006_df_outlier.py
	
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
# 004_df_handling.py
df.head();
df.info(); 
df.describe();
df.describe().T; 
df.shape;
df.sample(); 
df.columns();
df.describe; 
df.columns;
df.isnull().count()

#++++++++++++++
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
#++++++++++++++
    
    

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

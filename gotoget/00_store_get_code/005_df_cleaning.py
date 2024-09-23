# 005_df_cleaning.py
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



# 014_df_Encoding_obj_cols.py
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
  

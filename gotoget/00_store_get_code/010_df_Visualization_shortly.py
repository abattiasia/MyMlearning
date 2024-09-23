


# 010_df_Visualization_shortly.py

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



#008_df_load_sns_iris_titanic.py 
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

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#load.sns.titanic:
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

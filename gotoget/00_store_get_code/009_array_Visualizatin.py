

#009_array_Visualizatin.py

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


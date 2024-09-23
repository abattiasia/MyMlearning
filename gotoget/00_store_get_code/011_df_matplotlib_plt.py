
#011_df_matplotlib_plt.py

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

plt.scatter:
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
plt.show() 
##+++++++++++++++++

plt_hist(variable):   ##plot for all num_cols in df
plt.figure(figsize= (9,3))
plt.hist(dftrain[variable], bins=100)
plt.xlabel(variable)
plt.ylabel("frequency")
plt.title("{} distrubition with hist".format(variable))
plt.show()
num_cols =["Fare", "Age", "PassengerId"]
num_cols = df.select_dtypes(include=['number']).columns
for n in num_cols:
	plt_hist(n)
	
##+++++++++++++++++

#plt.pie:
gender_counts = df['gender'].value_counts()
## رسم المخطط الدائري
plt.figure(figsize=(8, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['ff9999','66b3ff'])
plt.title('Distribution of Gender')
plt.axis('equal')  ## Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
##+++++++++++++++++

#plt.pie_divided_col01_to_cats:
## تقسيم العملاء إلى فئات حسب مدة الاشتراك (tenure)
bins = [0, 12, 24, 48, 60, 72]
labels = ['0-12 months', '12-24 months', '24-48 months', '48-60 months', '60-72 months']
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels, right=False)
##+++++++++++++++++

## حساب توزيع العملاء حسب الفئات
tenure_group_counts = df['tenure_group'].value_counts().sort_index()
##+++++++++++++++++

## رسم المخطط الدائري
plt.figure( figsize =(8, 6))
plt.pie( tenure_group_counts, labels=tenure_group_counts.index, autopct='%1.1f%%', startangle=140, colors=['ff9999', '66b3ff', '99ff99', 'ffcc99', 'c2c2f0'])
plt.title(' Distribution of Customers by Tenure Group')
plt.axis('equal')  ## Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
##+++++++++++++++++

#plt.pie_2_cols_in_subplot:  
## حساب توزيع العملاء حسب خدمة الهاتف (PhoneService)
phone_service_counts = df['PhoneService'].value_counts()
##+++++++++++++++++

## حساب توزيع العملاء حسب الأمان الإلكتروني (OnlineSecurity)
online_security_counts = df['OnlineSecurity'].value_counts()
##+++++++++++++++++

## رسم المخطط الدائري لخدمة الهاتف
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.pie(phone_service_counts, labels=phone_service_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff'])
plt.title('Distribution of Customers by Phone Service')
plt.axis('equal')  ## Equal aspect ratio ensures that pie is drawn as a circle.
##+++++++++++++++++

## رسم المخطط الدائري للأمان الإلكتروني
plt.subplot(1, 2, 2)
plt.pie(online_security_counts, labels=online_security_counts.index, autopct='%1.1f%%', startangle=140, colors=['##99ff99', '##ffcc99', '##c2c2f0'])
plt.title('Distribution of Customers by Online Security')
plt.axis('equal')
plt.tight_layout()
plt.show()
##+++++++++++++++++

#plt.pie_3_cols_in_subplot:
## رسم المخطط الدائري لعمود StreamingTV
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
streaming_tv_counts = df['StreamingTV'].value_counts()
plt.pie(streaming_tv_counts, labels=streaming_tv_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff', '##99ff99'])
plt.title('Distribution of Customers by Streaming TV')
##+++++++++++++++++

## رسم المخطط الدائري لعمود StreamingMovies
plt.subplot(1, 3, 2)
streaming_movies_counts = df['StreamingMovies'].value_counts()
plt.pie(streaming_movies_counts, labels=streaming_movies_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff', '##99ff99'])
plt.title('Distribution of Customers by Streaming Movies')
##+++++++++++++++++

## رسم المخطط الدائري لعمود Contract
plt.subplot(1, 3, 3)
contract_counts = df['Contract'].value_counts()
plt.pie(contract_counts, labels=contract_counts.index, autopct='%1.1f%%', startangle=140, colors=['##ff9999', '##66b3ff', '##99ff99'])
plt.title('Distribution of Customers by Contract')
plt.show()
    
 
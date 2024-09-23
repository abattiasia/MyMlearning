
#012_df_seaborn_sns.py

sns.boxplot_features:{
	features = df.iloc[:,:9].columns.tolist()
	plt.figure(figsize=(18, 27))
	for i, col in enumerate(features):
		plt.subplot(6, 4, i*2+1)
		plt.subplots_adjust(hspace =.25, wspace=.3)
sns.boxplot(y = col, data = df, x="target", palette = ["blue", "yellow"]) }
##+++++++++++++++++

sns.countplot:
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
##++++++++++++++++++
	
#sns.histplot_col01_with_hue:
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
##++++++++++++++++++

#plt.pie_df.groupby:
## توزيع العقود بناءً على فئة StreamingTV
streaming_tv_contract_counts = df.groupby('StreamingTV')['Contract'].value_counts().unstack()
plt.figure(figsize=(10, 6))
for i, tv_option in enumerate(streaming_tv_contract_counts.index):
	plt.subplot(1, len(streaming_tv_contract_counts.index), i + 1)
	plt.pie(streaming_tv_contract_counts.loc[tv_option], labels=streaming_tv_contract_counts.columns, autopct='%1.1f%%', startangle=140)
	plt.title(f'Contracts for {tv_option} StreamingTV')
plt.tight_layout()
plt.show()
##++++++++++++++++++

## توزيع العقود بناءً على فئة StreamingMovies
streaming_movies_contract_counts = df.groupby('StreamingMovies')['Contract'].value_counts().unstack()
plt.figure(figsize=(10, 6))
for i, movies_option in enumerate(streaming_movies_contract_counts.index):
	plt.subplot(1, len(streaming_movies_contract_counts.index), i + 1)
	plt.pie(streaming_movies_contract_counts.loc[movies_option], labels=streaming_movies_contract_counts.columns, autopct='%1.1f%%', startangle=140)
	plt.title(f'Contracts for {movies_option} StreamingMovies')
plt.tight_layout()
plt.show()
##++++++++++++++++++

# sns.countplot_to_define_relation_between_2Cols:
plt.figure(figsize=(10, 5))
sns.countplot(x='heart_disease', hue='diabetes', data=df, palette=['darkturquoise', 'royalblue'])
plt.title('Count of Diabetes')
plt.xlabel('heart_disease')
plt.ylabel('Count')
plt.legend(title='diabetes', labels=['Not diabetes', 'diabetes'])
plt.show()

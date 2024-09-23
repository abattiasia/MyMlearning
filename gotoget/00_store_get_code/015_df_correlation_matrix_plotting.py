
# 015_df_correlation_matrix_plotting.py
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),cbar = True, annot =True,cmap='twilight_shifted_r')
plt.title('correlation matrix',fontdict={'fontsize': 20},fontweight ='bold')
plt.savefig("corr_matrix.png")

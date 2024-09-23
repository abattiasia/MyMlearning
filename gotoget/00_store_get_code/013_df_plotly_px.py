
#013_df_plotly_px.py
#plot for 2 cols. in df
fig = px.bar(
	df, x='name', y='price',  color='category', 
	title='Expected Profit by Subject and Level',
	labels={'subject': 'Subject', 'expected_profit': 'Expected Profit (USD)'},
	hover_data=['price', 'price', 'price']  ## Add more details on hover
    title_font_size=20,  ## Increase title font size
	xaxis_title_font_size=14,  ## Increase x-axis label font size
	yaxis_title_font_size=14,  ## Increase y-axis label font size
	hoverlabel=dict( bgcolor="white", font_size=12, font_family="Rockwell" )
)
fig.show()
  

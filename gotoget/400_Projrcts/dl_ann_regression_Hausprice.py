
from sklearn.metrics import r2_score

df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
df.dropna(inplace=True)

# Step 4: Encode Categorical Data
label_encoder = LabelEncoder()
df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])


# Step 5: Split the Data into Features (X) and Target (y)
X = df.drop('median_house_value', axis=1).values  # Features
y = df['median_house_value'].values  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)



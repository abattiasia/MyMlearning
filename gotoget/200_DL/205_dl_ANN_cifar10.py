
#205_dl_ANN_cifar10.py

#https//algoExpert.com

##Code:

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# 1. Load the cifar10 Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2. Preprocess the Data
# Normalize the data by scaling pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)  # 10 classes (0-9 digits)
y_test = to_categorical(y_test, 10)

# 3. Build the Model
model_ann = Sequential()

# 1 layers (input)
# Flatten the 28x28 images into 1D vectors of size 784
model_ann.add(Flatten(input_shape=(32, 32 ,3)))

# Add a fully connected hidden layer with 128 neurons and ReLU activation
model_ann.add(Dense(10, activation='relu'))

# Add a fully connected hidden layer with 128 neurons and ReLU activation
model_ann.add(Dense(10, activation='relu'))

# Add a fully connected hidden layer with 128 neurons and ReLU activation
model_ann.add(Dense(10, activation='relu'))

# Add the output layer with 10 neurons (one for each class) and softmax activation
model_ann.add(Dense(10, activation='softmax'))

# 4. Compile the Model
model_ann.compile(optimizer='adam',  loss='categorical_crossentropy',   metrics=['accuracy'])

# 5. Train the Model
model_ann.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# 6. Evaluate the Model on Test Data
test_loss, test_acc = model_ann.evaluate(X_test, y_test)
test_acc

# 7. Make Predictions
predictions = model_ann.predict(X_test[:5])
predictions.argmax(axis=1)  # axis=0 : row , axis=1 : column 


import matplotlib.pyplot as plt

# Display the first test image
plt.imshow(X_test[0]) #, cmap='color')
plt.title("First Test Image")
plt.show()

#Make Prediction for the first test image
prediction = model_ann.predict(X_test[:3]) #.reshape(3, 32, 32))

# Show the predicted label and the actual label
print('Predicted label:', prediction.argmax())
print('Actual label:', y_test[0].argmax())

# Print the model summary
model_ann.summary()


# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.callbacks import EarlyStopping
# # # Load the dataset
# # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# #  # Normalize the images to [0, 1] range
# # x_train, x_test = x_train / 255.0, x_test / 255.0
#  # Build a simple ANN model

# # model = Sequential([
# #      Flatten(input_shape=(32, 32, 3)),
# #      Dense(128, activation='relu'),
# #      Dense(10, activation='softmax')
# # ])
# # Compile the model
# ###model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# ###model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  #loss='sparse_categorical_crossentropy'
# # Define early stopping callback
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
# # Train the model with early stopping
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, callbacks=[early_stopping])


# Evaluate the model
test_loss, test_acc = model_cnn.evaluate(X_test, y_test)
test_acc


# Plot training and validation accuracy/loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
# Display the first 5 test images and predicted labels
predictions = model_cnn.predict(X_test[:5])

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i])
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Pred: {class_names[predictions[i].argmax()]}")
plt.show()






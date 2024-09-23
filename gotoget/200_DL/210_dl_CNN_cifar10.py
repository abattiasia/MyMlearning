
# 210_dl_CNN_cifar10.py
#hallo
#https//algoExpert.com

##Code:

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from tensorflow.keras import layers

# 1. Load the cifar10 Dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2. Preprocess the Data
# Normalize the data by scaling pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)  # 10 classes (0-9 digits)
y_test = to_categorical(y_test, 10)



#  # Build a simple CNN model
# Initialize the Sequential model

from tensorflow.keras import layers
model_cnn = Sequential()  #models.

# Add layers step-by-step using the add() method
model_cnn.add(layers.InputLayer(input_shape=(32, 32, 3)))  # Input layer for 32x32x3 color images
model_cnn.add(layers.Conv2D(32, (3, 3), activation='relu'))  # First Conv2D layer
model_cnn.add(layers.MaxPooling2D((2, 2)))  # First MaxPooling layer
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Second Conv2D layer
model_cnn.add(layers.MaxPooling2D((2, 2)))  # Second MaxPooling layer
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Third Conv2D layer

model_cnn.add(layers.Flatten())  # Flatten the feature maps
model_cnn.add(layers.Dense(64, activation='relu'))  # Fully connected layer
model_cnn.add(layers.Dense(10, activation='softmax'))  # Output layer for 10 classes

# Compile the model
model_cnn.compile(optimizer='adam',
              loss='categorical_crossentropy',   #sparse_categorical_crossentropy
              metrics=['accuracy'])

# Print the model summary
model_cnn.summary()

## Train the model
#model_ann.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)   ->>> ANN

#reshaped_X_train = X_train.reshape(3, 32, 32).transpose(1, 2, 0) 
#X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
#y_test = y_test.reshape((y_test.shape[0], 32, 32, 3))

history = model_cnn.fit(X_train, y_train, epochs=10, validation_split=0.2)


y_train.shape

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

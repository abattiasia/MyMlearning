# DL_CNN
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

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
              loss='categorical_crossentropy',   #sparse_categorical_crossentropy  ..> only wenn no encoding for y_train
              metrics=['accuracy'])

history = model_cnn.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Train the model
#model_ann.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)   ->>> ANN

#reshaped_X_train = X_train.reshape(3, 32, 32).transpose(1, 2, 0) 
#X_train = X_train.reshape((X_train.shape[0], 32, 32, 3))
#y_test = y_test.reshape((y_test.shape[0], 32, 32, 3))



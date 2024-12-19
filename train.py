import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Specify dataset paths
train_data_path = r'C:\Users\deban\Downloads\pneumonia detection\train'
test_data_path = r'C:\Users\deban\Downloads\pneumonia detection\test'
# Check if the dataset paths exist and have the correct structure
if not os.path.exists(train_data_path):
    raise FileNotFoundError(f"Training data path not found: {train_data_path}")
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Testing data path not found: {test_data_path}")

# Ensure that the directory structure is correct
if not any(os.listdir(train_data_path)):
    raise FileNotFoundError("Training data directory is empty or improperly structured.")
if not any(os.listdir(test_data_path)):
    raise FileNotFoundError("Testing data directory is empty or improperly structured.")

# Data augmentation for training data
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_data_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='binary'
)

# Data preprocessing for testing data
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_data = test_datagen.flow_from_directory(
    test_data_path, 
    target_size=(224, 224), 
    batch_size=32, 
    class_mode='binary'
)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, validation_data=test_data, epochs=10)

# Save the model for later use
model.save('pneumonia_model.h5')

# Evaluate the model
predictions = (model.predict(test_data) > 0.5).astype(int)
cm = confusion_matrix(test_data.classes, predictions)

# Plot the confusion matrix
plt.matshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

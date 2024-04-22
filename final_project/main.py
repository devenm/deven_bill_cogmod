import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras import layers
#Data Preprocessing
print('Loading data...')
data_train_path = 'data/train'
data_test_path = 'data/test'
data_val_path = 'data/val'

img_width = 180
img_height =180 

data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width, img_height),
    batch_size=32,
    validation_split=False)

data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    image_size=(img_height,img_width),
    shuffle=False,
    batch_size=32,
    validation_split=False)

data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    image_size=(img_height,img_width),
    batch_size=32,
    shuffle=False,
    validation_split=False)

data_cat = data_train.class_names
print('Data loaded.')

#Setting up deep learning model and subsequent layers
print('Setting up base model...')

#Compile and fit models
def eval_sgd():
    model = keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_cat))])

    lrs = [0.0001, 0.001, 0.01]
    for lr in lrs:
        opt = keras.optimizers.SGD(learning_rate=lr)

        model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        epochs_size = 20
        history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)

        #Plot performance
        print('Graphing accuracies...')
        epochs_range = range(epochs_size)
        plt.figure(figsize=(8,8))
        plt.subplot(1,2,1)
        plt.plot(epochs_range,history.history['accuracy'],label = 'Training Accuracy')
        plt.plot(epochs_range, history.history['val_accuracy'],label = 'Validation Accuracy')
        plt.title('Accuracy')
        plt.subplot(1,2,2)
        plt.plot(epochs_range,history.history['loss'],label = 'Training Loss')
        plt.plot(epochs_range, history.history['val_loss'],label = 'Validation Loss')
        plt.title('Loss')

def eval_adam():

    model = keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_cat))])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    epochs_size = 20
    history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)

    #Plot performance
    print('Graphing accuracies...')
    epochs_range = range(epochs_size)
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(epochs_range,history.history['accuracy'],label = 'Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'],label = 'Validation Accuracy')
    plt.title('Accuracy')
    plt.subplot(1,2,2)
    plt.plot(epochs_range,history.history['loss'],label = 'Training Loss')
    plt.plot(epochs_range, history.history['val_loss'],label = 'Validation Loss')
    plt.title('Loss')


eval_sgd()
eval_adam()
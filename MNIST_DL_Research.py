# -*- coding: utf-8 -*-
"""
Created on Mon May 18 20:24:01 2020

@author: Robert
"""
#%% ----- Simple Neural Network exploring MNIST dataset -----

# Capture start time
import time as time
t = time.time()

# Import tensorflow and initialize Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import TensorBoard
import pandas as pd
import numpy as np

# load MNIST data set
(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()

# Transform data into the format Keras expects. 

# Divide the image data by 255 in order to normalize it into 0-1 range, 
# after converting it into floating point values.

train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000, 784)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255

# Convert the 0-9 labels into "one-hot" format
# Ex. The label vector representing the number 1 would be 
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

# Visualy display MNIST images for insight

import matplotlib.pyplot as plt
#Make nice looking plots
#import seaborn as sns
#sns.set()

# Plot 9 images from dataset randomly
import random 
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# generate random number
	num = random.randrange(60000)
	#Convert label back to a number
	label = train_labels[num].argmax(axis=0)
	#Print the one-hot array of this sample's label
	print("Label", label, ", One-Hot Array:", train_labels[num])
	#Reshape the 768 values to a 28x28 image
	image = train_images[num].reshape([28,28])
	plt.title('Sample: %d  Label: %d' % (num, label))
	plt.subplots_adjust(hspace=0.4)
	plt.imshow(image, cmap=plt.get_cmap('gray_r'))

#%% ----- Defining the Network -----

# A Sequential model is appropriate for a plain stack of layers where each layer 
# has exactly one input tensor and one output tensor.

# The rectified linear activation function is a piecewise linear function that 
# will output the input directly if is positive, otherwise, it will output zero. 
# It has become the default activation function for many types of neural networks 
# because a model that uses it is easier to train and often achieves better performance.

# Why softmax?

# The values coming out of our matrix operations can have large, and negative
# values. We would like our solution vector to be conventional probabilities that
# sum to 1.0. An effective way to normalize our outputs is to use the popular
# Softmax function.

# Softmax converts a real vector to a vector of categorical probabilities.
# Softmax is often used as the activation for the last layer of a classification 
# network because the result could be interpreted as a probability distribution.

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

#View description of resulting model
model.summary()

#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Setup optimizer, loss function and compile model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Add TensorBoard callback to activate logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='TB_logDIR', histogram_freq=1)

#%% ----- Training the model -----

# Here we'll do 10 epochs with a batch size of 100. 
# Keras is slower, and if we're not running on top of a GPU-accelerated Tensorflow 
# this can take a fair amount of time (that's why I've limited it to just 10 epochs.)

t_m1 = time.time()

history = model.fit(train_images, train_labels,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels),
                    callbacks=[tensorboard_callback])

model1_training_time = time.time() - t_m1

#%% ----- Evaluate Model -----

score = model.evaluate(test_images, test_labels, verbose=0)

# Initialize list of lists for Summary comparing Models
data = [['Simple Model 1', score[0], score[1], model1_training_time]] 
  
# Create the Summary DataFrame 
df_summary = pd.DataFrame(data, columns = ['Model Name', 'Test Loss', 'Test Accuracy', 'Training Time']) 

# Visualize performance

plt.figure(2)

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model 1 accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model 1 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Visualy display MNIST images Model got wrong for insight
 
fig_num = 0
for x in range(1000):
    if fig_num < 9:
        test_image = test_images[x,:].reshape(1,784)
        predicted_cat = model.predict(test_image).argmax()
        label = test_labels[x].argmax()
        if (predicted_cat != label):        
            plt.subplot(330 + 1 + fig_num)
            #plt.figure(fig_num)
            plt.title('Prediction: %d Label: %d' % (predicted_cat, label))
            plt.subplots_adjust(hspace=0.4)
            plt.imshow(test_image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
            #plt.show()
            fig_num = fig_num + 1

#%% ----- Defining a wider Network -----

# Increase Neurons from 64 to 512          
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#%% ----- Training the wider model -----

t_m2 = time.time()

history = model.fit(train_images, train_labels,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))

model2_training_time = time.time() - t_m2

#%% ----- Evaluate the wider Model -----

score = model.evaluate(test_images, test_labels, verbose=0)

# Create Temp df to add model metrics
df_temp = pd.DataFrame([['Wider Model 2', score[0], score[1], model2_training_time]], 
                       columns = ['Model Name', 'Test Loss', 'Test Accuracy', 'Training Time'])

# Append temp dataframe to summary dataframe
df_summary = df_summary.append(df_temp, ignore_index=True)

# Visualize performance

plt.figure(4)

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Wider Model 2 accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Wider Model 2 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%% ----- Defining a deeper and wider Network -----

# Add another layer with 512 neurons            
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#%% ----- Training the deeper and wider model -----

t_m3 = time.time()

history = model.fit(train_images, train_labels,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))

model3_training_time = time.time() - t_m3

#%% ----- Evaluate the deeper and wider Model -----

score = model.evaluate(test_images, test_labels, verbose=0)

# Create Temp df to add model metrics
df_temp = pd.DataFrame([['Deeper and Wider Model 3', score[0], score[1], model3_training_time]], 
                       columns = ['Model Name', 'Test Loss', 'Test Accuracy', 'Training Time'])

# Append temp dataframe to summary dataframe
df_summary = df_summary.append(df_temp, ignore_index=True)

# Visualize performance

plt.figure(5)

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Deeper and Wider Model 3 accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Deeper and Wider Model 3 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%% ----- Defining adding dropout to a deeper and wider Network -----

#Add dropout layer between layers            
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#%% ----- Training the deeper and wider model -----

t_m4 = time.time()

history = model.fit(train_images, train_labels,
                    batch_size=100,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))

model4_training_time = time.time() - t_m4

#%% ----- Evaluate the Model with dropout -----

score = model.evaluate(test_images, test_labels, verbose=0)

# Create Temp df to add model metrics
df_temp = pd.DataFrame([['Dropout Model 4', score[0], score[1], model4_training_time]], 
                       columns = ['Model Name', 'Test Loss', 'Test Accuracy', 'Training Time'])

# Append temp dataframe to summary dataframe
df_summary = df_summary.append(df_temp, ignore_index=True)

# Visualize performance

plt.figure(6)

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Dropout Model 4 accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Dropout Model 4 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#%% ----- Using a Convolutional Neural Network -----

# Need to shape data differently than before. 
# Treading data as 2D (28x28) images not flattened 784 pixels

from tensorflow.keras import backend as K

# "1" indicates a single color channel, grayscale only. use "3" for color
if K.image_data_format() == 'channels_first':
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28, 28)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
    test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Convert train and test labels to categorical in one-hot format 
train_labels = tf.keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = tf.keras.utils.to_categorical(mnist_test_labels, 10)

# Define CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# 64 3x3 kernels
model.add(Conv2D(64, (3, 3), activation='relu'))
# Reduce by taking the max of each 2x2 block
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout to avoid overfitting
model.add(Dropout(0.25))
# Flatten the results to one dimension for passing into our final layer
model.add(Flatten())
# A hidden layer to learn with
model.add(Dense(128, activation='relu'))
# Another dropout
model.add(Dropout(0.5))
# Final categorization from 0-9 with softmax
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

t_m5 = time.time()

history = model.fit(train_images, train_labels,
                    batch_size=32,
                    epochs=10,
                    verbose=2,
                    validation_data=(test_images, test_labels))

model5_training_time = time.time() - t_m5

score = model.evaluate(test_images, test_labels, verbose=0)

# Create Temp df to add model metrics
df_temp = pd.DataFrame([['CNN Model 5', score[0], score[1], model5_training_time]], 
                       columns = ['Model Name', 'Test Loss', 'Test Accuracy', 'Training Time'])

# Append temp dataframe to summary dataframe
df_summary = df_summary.append(df_temp, ignore_index=True)

# Visualize performance

plt.figure(7)

plt.subplot(211)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN Model 5 accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN Model 5 loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

fig_num = 0
for x in range(1000):
    if fig_num < 9:
        test_image = test_images[x,:].reshape(1, 28, 28, 1)
        predicted_cat = model.predict(test_image).argmax()
        label = test_labels[x].argmax()
        if (predicted_cat != label):        
            plt.subplot(330 + 1 + fig_num)
            #plt.figure(fig_num)
            plt.title('CNN Prediction: %d Label: %d' % (predicted_cat, label))
            plt.subplots_adjust(hspace=0.4)
            plt.imshow(test_image.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
            #plt.show()
            fig_num = fig_num + 1
            
#%% Display Summary Dataframe
print(df_summary)

elapsed = time.time() - t
elapsed = elapsed//60
print('Program Complete, Total Time Elapsed: %s minutes' % elapsed)
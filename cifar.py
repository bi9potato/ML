#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:55:31 2021

@author: prime
"""
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
gpu = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(gpu[0], True) 

cifar10 = tf.keras.datasets.cifar10 
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data() 
train_images, test_images = train_images / 255.0, test_images / 255.0 

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, input_shape=(32, 32, 3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Conv2D(64, 3, activation='relu'), 
    tf.keras.layers.MaxPooling2D(), 
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(10,  activation='softmax') 
    ])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 
#%%
history = model.fit(train_images, train_labels, epochs=5, 
                    
          validation_data=(test_images, test_labels)) 

#%% 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss'])
#%%
layers = model.layers
#%% 
c0 = layers[0]
#%%
w0 = c0.get_weights() # function
#%% 
u0 = c0.get_weights # attribution
#%% 
index = 9
x = train_images[index]
plt.imshow(x)
#%%
x0 = x.reshape(1, 32, 32, 3) 
#%%
x1 = layers[0](x0)
#%%
x1 = x1.numpy()
#%% 
for i in range(32):
    plt.subplot(4, 8, i + 1) 
    plt.imshow(x1[0, :, :, i]) 
    plt.xticks([]) 
    plt.yticks([]) 
    
#%%
x2 = layers[1](x1) 
x2 = x2.numpy()
#%%
plt.figure() 
for i in range(32):
    plt.subplot(4, 8, i + 1) 
    plt.imshow(x2[0, :, :, i]) 
    plt.xticks([]) 
    plt.yticks([]) 
    
#%%
c2 = layers[2]
w2 = c2.get_weights() 
#%%
x3 = layers[2](x2).numpy() 
#%%
plt.figure() 
for i in range(64):
    plt.subplot(8, 8, i + 1) 
    plt.imshow(x3[0, :, :, i]) 
    plt.xticks([]) 
    plt.yticks([]) 
#%%
x4 = layers[3](x3).numpy() 
plt.figure() 
for i in range(64):
    plt.subplot(8, 8, i + 1) 
    plt.imshow(x4[0, :, :, i]) 
    plt.xticks([]) 
    plt.yticks([]) 
    
#%% 
x5 = layers[4](x4).numpy() 
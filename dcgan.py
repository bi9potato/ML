#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:06:40 2021

@author: prime
"""

import tensorflow as tf
import matplotlib.pyplot as plt 
import os 
import time 

gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True) 


def make_generator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100, )), 
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.LeakyReLU(), 
        tf.keras.layers.Reshape((7, 7, 256)), 
        tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False), 
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.LeakyReLU(), 
        tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False), 
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.LeakyReLU(), 
        tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ])
    return model 

def make_discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]), 
        tf.keras.layers.LeakyReLU(), 
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'), 
        tf.keras.layers.LeakyReLU(), 
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(1)
        ])
    return model 

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) 

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) 
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) 
    total_loss = real_loss + fake_loss 
    return total_loss 

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output) 

generator = make_generator_model() 
discriminator = make_discriminator_model() 
generator_optimizer = tf.keras.optimizers.Adam(1e-4) 
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4) 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt') 
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                 discriminator_optimizer=discriminator_optimizer, 
                                 generator=generator, 
                                 discriminator=discriminator) 
noise_dim = 100 
num_examples_to_generate = 16 
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function 
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: 
        generated_images = generator(noise, training=True) 
        
        real_output = discriminator(images, training=True) 
        fake_output = discriminator(generated_images, training=True) 
        
        gen_loss = generator_loss(fake_output) 
        disc_loss = discriminator_loss(real_output, fake_output) 
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) 
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables) 
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) 
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables)) 
    
def generate_and_save_images(model, epoch, test_input, image_path='./generated_image'):
    # Notice `training` is set to False. 
    # This is so all layers run in inference mode (batchnorm). 
    predictions = model(test_input, training=False) 
    fig = plt.figure(figsize=(8, 8)) 
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1) 
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray') 
        plt.axis('off') 
    plt.savefig(os.path.join(image_path,'image_at_epoch_{:04d}.png'.format(epoch)) )
    plt.close(fig) 

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time() 
        
        for image_batch in dataset:
            train_step(image_batch) 
            
        # Produce images for the gif as you go 
        if epoch % 20 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
        # Save the model every 200 epochs 
        if (epoch + 1) % 200 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix) 
        print("Time for epoch {} is {} sec.".format(epoch + 1, time.time() - start)) 
    generate_and_save_images(generator, epochs, seed) 

if __name__ == '__main__':
    EPOCHS = 100
    BUFFER_SIZE = 60000 
    BATCH_SIZE = 128 
    
    try:
        image_path = './generated_image'
        os.makedirs(image_path) 
    except FileExistsError:
        pass 
    
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data() 
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images- 127.5) / 127.5 # Normalize the image to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE) 
    
    train(train_dataset, EPOCHS) 
    
    
    


        


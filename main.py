import tensorflow as tf
import numpy as np
import os
import time
import pickle
from glob import glob

import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.layers import *    
from tensorflow.keras.optimizers import Adam

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    #normalize to [-1,1]
    img = (img - 127.5) / 127.5
    return img

def tf_dataset(images_path, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images_path)
    dataset = dataset.shuffle(buffer_size = 300)
    dataset = dataset.map(load_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def build_generator():
    model = tf.keras.Sequential(name = 'generator')
    model.add(Dense(4*4*1024, use_bias = False, input_shape = (100,)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Reshape((4,4,1024)))
    
    #8x8
    model.add(Conv2DTranspose(512, (5,5), strides = (2,2), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    #16x16
    model.add(Conv2DTranspose(256, (5,5), strides = (2,2), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    #32x32
    model.add(Conv2DTranspose(128, (5,5), strides = (2,2), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    #64x64x3
    model.add(Conv2DTranspose(3, (5,5), strides = (2,2), padding = 'same', use_bias = False,
                              activation = 'tanh'))
    return model

def build_discriminator():
    model = tf.keras.Sequential(name = 'discriminator')
    
    model.add(Conv2D(64, (4,4), (2,2), padding = 'same', use_bias = False, 
                     input_shape= (64,64,3)))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (4,4), (2,2), padding = 'same', use_bias = False))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    model.add(Conv2D(256, (4,4), (2,2), padding = 'same', use_bias = False))
    model.add(LeakyReLU())
    model.add(Dropout(0.2))
    
    # model.add(Conv2D(512, (4,4), (2,2), padding = 'same', use_bias = False))
    # model.add(LeakyReLU())
    # model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1))
    
    return model

def discriminator_loss(real, fake):
    real_loss = cross_entropy(tf.ones_like(real), real)

    fake_loss = cross_entropy(tf.zeros_like(fake), fake)
    total_loss = fake_loss + real_loss
    return total_loss

def generator_loss(fake):
    return cross_entropy(tf.ones_like(fake), fake)

@tf.function
def train_step(batch_size, images):
    noise = tf.random.normal((batch_size, 100))
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training = True)
        
        real = discriminator(images, training = True)
        fake = discriminator(generated_images, training = True)
        
        gen_loss = generator_loss(fake)
        disc_loss = discriminator_loss(real, fake)
        
    gen_gradient = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradient = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gen_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradient, discriminator.trainable_variables))
    
    return (gen_loss, disc_loss)
    
def generate_and_save_images(path, model, epoch, seed):
    if not os.path.exists(path):
        os.makedirs(path)
    
    generated_images = model(seed, training=False)
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = generated_images[i]*127.5 + 127.5
        img = img.numpy().astype('uint8')
        img = Image.fromarray(img)
        plt.imshow(img)
        plt.axis('off')
        
    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(path, epoch))
    plt.clf()

def train(dataset, epochs):
    plt.figure(figsize = (10,10))
    
    for epoch in range(epochs):
        start = time.time()
        
        for i, image_batch in enumerate(dataset):
            losses = train_step(BATCH_SIZE, image_batch)
            gen_hist.append(losses[0])
            disc_hist.append(losses[1])
            progbar.update(i+1)
            
        generate_and_save_images(OUTPUT_PATH, generator, epoch, seed)
        
        if (epoch + 1) % 25 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        print('Generator loss: {:2f} Discriminator loss: {:.2f}'.format(losses[0], losses[1]))

if __name__ == '__main__':
    EPOCHS = 300
    BATCH_SIZE = 32
    seed = tf.random.normal((16, 100)) #seed to generate example images to illustrate training process
    DATA_PATH = 'transformed_data/*'
    OUTPUT_PATH = './generated_images'
    CKPT_DIR = './checkpoints'
    
    images_path = glob(DATA_PATH)
    
    images = tf_dataset(images_path, BATCH_SIZE)
    
    generator = build_generator()
    discriminator = build_discriminator()
    generator.summary()
    discriminator.summary()
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    generator_optimizer = Adam(learning_rate=0.0002,beta_1=0.7)
    discriminator_optimizer = Adam(learning_rate=0.0002,beta_1=0.7)
    
    checkpoint_prefix = os.path.join(CKPT_DIR, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                      discriminator_optimizer=discriminator_optimizer,
                                      generator = generator,
                                      discriminator= discriminator)
    
    gen_hist = []
    disc_hist = []
    
    progbar = tf.keras.utils.Progbar(len(list(images.as_numpy_iterator())))
    
    #checkpoint.restore(tf.train.latest_checkpoint(CKPT_DIR))
    train(images, EPOCHS)
    
    with open('pkm_gen_loss.pkl', 'wb') as f:
        pickle.dump(gen_hist, f)
    with open('pkm_disc_loss.pkl', 'wb') as f:
        pickle.dump(disc_hist, f)    

    






    
    
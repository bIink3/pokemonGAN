from main import *
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def generate_pokemon():
    seed = tf.random.normal((1,100))
    output = generator(seed, training = False)
    img = output[0]*127.5 + 127.5
    img = img.numpy().astype('uint8')
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.axis('off')

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = Adam(learning_rate=0.0002,beta_1=0.7)
discriminator_optimizer = Adam(learning_rate=0.0002,beta_1=0.7)

CKPT_DIR = './checkpoints'
checkpoint_prefix = os.path.join(CKPT_DIR, 'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator = generator,
                                discriminator= discriminator)

checkpoint.restore(tf.train.latest_checkpoint(CKPT_DIR))

generate_pokemon()
### import library
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from IPython.display import clear_output

from tensorflow_examples.models.pix2pix import pix2pix

tfds.disable_progress_bar()
autotune = tf.data.AUTOTUNE


### fix config
# Define the standard image size.
orig_img_size = (512, 512)
# Size of the random crops to be used during training.
input_img_size = (1, 512, 512, 3)

# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

### parameters
buffer_size = 256
batch_size = 1
EPOCHS = 1000

directory = 'monet/'
version = 'v2/'
weight_name = '1000_epochs'
version_dir = './weight_files/' + directory + version

per_save = 50

if EPOCHS%per_save==0:
    num_chpt=EPOCHS//per_save
else:
    num_chpt = EPOCHS//per_save+1

print('\n')
print("############################\n")

print(f"save checkpoint per {per_save} epochs")

print(f'It will save {num_chpt} chpt_files')

print('\n')
print("############################\n")


### load dataset
# original_picture
# ds_ori = tf.keras.utils.image_dataset_from_directory('../images/real_hockney', batch_size=batch_size, image_size = orig_img_size, shuffle=False)
ds_ori = tf.keras.utils.image_dataset_from_directory('../images/real_hockney', batch_size=batch_size, image_size = orig_img_size)

# pixel_art
# ds_tar = tf.keras.utils.image_dataset_from_directory('../images/hockney', batch_size=batch_size, image_size = orig_img_size, shuffle=False)
ds_tar = tf.keras.utils.image_dataset_from_directory('../images/monet', batch_size=batch_size, image_size = orig_img_size)

# custom_testset
ds_cus = tf.keras.utils.image_dataset_from_directory('../images/origin', batch_size=batch_size, image_size = orig_img_size)


### preprocessing
# normalize_image
def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0

### train dataset preprocessing
def preprocess_train_image(img, label):
    # # to grayscale
    # img = tf.image.rgb_to_grayscale(img, name=None)
    # # to rgb
    # img = tf.image.grayscale_to_rgb(img)
    # # Random flip
    # img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, [*orig_img_size])
    # Random crop to 256X256
    img = tf.image.random_crop(img, size=[*input_img_size])
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


### test dataset preprocessing
def preprocess_test_image(img, label):
    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
#     img = tf.image.resize(img, [input_img_size[1], input_img_size[2]])

    img = normalize_img(img)
    return img


# ## preprocessing
# ds_ori_t = ds_ori.map(preprocess_train_image, num_parallel_calls=autotune).cache()

# ds_tar_t = ds_tar.map(preprocess_train_image, num_parallel_calls=autotune).cache()
ds_ori_t = ds_ori.map(preprocess_train_image, num_parallel_calls=autotune).cache().shuffle(buffer_size)

ds_tar_t = ds_tar.map(preprocess_train_image, num_parallel_calls=autotune).cache().shuffle(buffer_size)

ds_cus_t = ds_cus.map(preprocess_train_image, num_parallel_calls=autotune).cache().shuffle(buffer_size)



# generator and discriminator
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


### Load latest checkpoint
checkpoint_path = version_dir


### save models
exist_dir = os.path.exists(checkpoint_path)

# create directory
if not exist_dir:
    os.makedirs(checkpoint_path)

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, 
                                        checkpoint_path, 
                                        checkpoint_name=weight_name, 
                                        max_to_keep=num_chpt)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('\n')
    print("############################\n")
    print ('Latest checkpoint restored!!')



# def generate_images(model, test_input):
#     prediction = model(test_input)

#     plt.figure(figsize=(12, 12))

#     display_list = [test_input[0], prediction[0]]
#     title = ['Input Image', 'Predicted Image']

#     for i in range(2):
#         plt.subplot(1, 2, i+1)
#         plt.title(title[i])
#         # getting the pixel on values between [0, 1] to plot it.
#         plt.imshow(display_list[i] * 0.5 + 0.5)
#         plt.axis('off')
#     plt.show()


@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


print('\n')
print("############################\n")

print(f"strat training!!!")

start = time.time()
start_time = datetime.fromtimestamp(start)
print(f'start training - time : {start_time}')
time_ = time.time()

print('\n')
print("############################\n")

for epoch in range(EPOCHS):
    epoch_time = time.time()
    
    n = 0
    for image_x, image_y in tf.data.Dataset.zip((ds_ori_t, ds_tar_t)):
        train_step(image_x, image_y)
        if n % 10 == 0:
            print ('.', end='')
        n+=10

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    # generate_images(generator_g, sample_horse)
    epoch_ = epoch+1

    fin_time = time.time()
    take_time = fin_time - epoch_time
    time_ += take_time
    
    print(f'finished : {epoch_} / {EPOCHS} epochs, time taken : {take_time} sec')

    if (epoch_) % per_save == 0:
        # save checkpoint
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                     ckpt_save_path))

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                  time_))

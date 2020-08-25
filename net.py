import os
import pylab as pl
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow_probability as tfp

from natsort import natsorted
from random import randrange

keras.backend.set_floatx("float32")

def get_model():
    # input batch shape will be (1, 180, 320, 15)
    inp = keras.Input(shape=(180, 320, 12))

    #x = keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(180, 320, 3), pooling='avg')(inp)

    enc = keras.Sequential([layers.Conv2D(32, 9, activation='swish'),
                            layers.Conv2D(64, 5, activation='swish'),
                            layers.Conv2D(128, 3, strides=2, activation='swish'),
                            layers.Conv2D(128, 3, activation='swish'),
                            layers.Conv2D(128, 3, activation='swish'),
                            layers.Conv2D(256, 3, strides=2, activation='swish'),
                            layers.Conv2D(256, 3, activation='swish'),
                            layers.Conv2D(256, 3, activation='swish')], name='encoder')(inp)

    dec = keras.Sequential([layers.Conv2DTranspose(256, 3, activation='swish'),
                            layers.Conv2DTranspose(256, 3, activation='swish'),
                            layers.Conv2DTranspose(256, 3, strides=2, activation='swish'),
                            layers.Conv2DTranspose(128, 3, activation='swish'),
                            layers.Conv2DTranspose(128, 3, activation='swish'),
                            layers.Conv2DTranspose(128, 3, strides=2, activation='swish'),
                            layers.Conv2DTranspose(64, 5, activation='swish'),
                            layers.Conv2DTranspose(32, 9, activation='swish'),
                            layers.Conv2DTranspose(3, 2)], name='decoder')(enc)

    return keras.Model(inputs=inp, outputs=dec)

model = get_model()
model.load_weights("weights.h5")
model.summary()

def get_images():
    index = randrange(175) # 180 frames minus 5

    files = natsorted(os.listdir("images"))

    images = []
    for i in range(5):
        filepath = "images/"+files[index+i]
        image = tf.image.decode_jpeg(tf.io.read_file(filepath))
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (180, 320))

        images.append(image)

    return images

images = get_images()

loss_met = keras.metrics.Mean()
logits_met = keras.metrics.Mean()

def predict(images):
    image_input = tf.expand_dims(tf.concat(images[:-1], -1), 0)
    label = tf.expand_dims(images[-1], 0)

    logitses = []
    for i in range(10):
        logits = model(image_input)
        logits = tf.clip_by_value(logits, 0., 1.)
        logitses.append(logits)
        image_input = tf.concat((image_input[0,...,3:], logits[0]), -1)
        image_input = tf.expand_dims(image_input, 0)

    display = images + logitses

    for i in range(len(display)):
        pl.imshow(tf.squeeze(display[i]))
        pl.draw()
        pl.pause(1)

    pl.imshow(tf.squeeze(logits))
    pl.show()

predict(images)

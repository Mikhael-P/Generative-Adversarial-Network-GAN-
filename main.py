from os import mkdir
from keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from models import build_discriminator, build_generator
from train import train
from utils import save_imgs
from config import args



data = mnist.load_data()
path = "./save_images"
img_shape = (args.img_rows, args.img_col, args.channels)

epochs = 15000
save_interval = 1000
lr=0.0002
beta=0.5
optimizer = Adam(lr, beta)

# Setting the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, 
                      metrics = ['accuracy'])

# Setting the generator
generator = build_generator(img_shape)
generator.compile(loss = 'binary_crossentropy', optimizer = optimizer)

# print()

# Noise
z = Input(shape=(100,))
img = generator(z)


discriminator.trainable = False


valid = discriminator(img) # Validity check on the generated image


combined = Model(z, valid)
combined.compile(loss = 'binary_crossentropy', optimizer = optimizer)

for epoch in range(epochs):
    train(epoch, data, generator, discriminator, combined, batch_size = 128, save_interval = save_interval)
    if epoch % save_interval == 0:
        save_imgs(epoch, generator, path)
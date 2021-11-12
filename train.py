import numpy as np


def train(epoch, data, generator, discriminator, combined, batch_size = 128, save_interval = 500):

    # load the dataset
    (X_train, _), (_, _) = data

    #Convert to float and rescale -1 to 1 (can also do 0 to 1)

    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    # Add channels dimension. As the input to our gen and discr. has a shape a shape 28x28x1

    X_train = np.expand_dims(X_train, 3)

    half_batch = int(batch_size / 2)

    #for epoch in range(epochs):
    # Train the discriminator

    # Select a random half batch of real images
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    imgs = X_train[idx]

    noise = np.random.normal(0, 1, (half_batch, 100))

    # Generate a half batch of fake images
    gen_imgs = generator.predict(noise)

    # Train the discriminator on real and fake images separately
    # Research showed that separate training is more effective
    d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

    #take average loss from real and fake images

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator

    #Create noise vectors as input for generator
    #Create as many noise vectors as defined by the batch size
    # based on normal distribution, output will be of size (batch size, 100) 

    noise = np.random.normal(0, 1, (batch_size, 100))

    # The generator wants the discriminator to label the generated samples as valid (ones)
    # This is where the generator is trying to trick discriminator into believing th
    # generated image is true (hence value of 1 for y)
    valid_y = np.array([1] * batch_size) # image probability 

    # Generator is part of combined where it got directly linked with the discriminator
    # Train the generator with noise as x and 1 as y.
    # Again, 1 as the output as it is adversarial and if generator did a great job of 
    # folling the discriminator then the output would be 1 (true)
    g_loss = combined.train_on_batch(noise, valid_y)

    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1],
                                                            g_loss))
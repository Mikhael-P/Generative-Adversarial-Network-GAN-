import numpy as np
import matplotlib.pyplot as plt

def save_imgs(epoch, generator, path):
  r, c = 5, 5
  noise = np.random.normal(0, 1, (r * c, 100))
  gen_imgs = generator.predict(noise)

  #Rescale images 0-1
  gen_imgs = 0.5 * gen_imgs + 0.5

  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
    for j in range(c):
      axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap = 'gray')
      axs[i,j].axis('off')
      cnt += 1

  fig.savefig(path + "/mnist_%d.png" % epoch)
  plt.close()
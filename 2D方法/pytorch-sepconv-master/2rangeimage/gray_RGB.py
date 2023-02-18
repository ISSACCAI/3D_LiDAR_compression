import torch
import random
import numpy as np
from PIL import Image

imagePath = 'C:/Users/Cauli/Desktop/44/oneone.png'
img2=Image.open(imagePath)
img2=np.array(img2)
img = np.expand_dims(img2, 0)
img_once=np.stack((img, img), axis=3)
img1 = np.expand_dims(img, 3)
# img_once1=np.stack((img_once, img1), axis=0)
img_once1 = np.concatenate([img_once, img1], axis=3)
img_two = np.squeeze(img_once1, axis=0)

im = Image.fromarray(img_two).convert('RGB')
im.save("C:/Users/Cauli/Desktop/44/twotwo.png")
print(0)


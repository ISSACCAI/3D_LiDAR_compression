import torch
import random
import numpy as np
from PIL import Image

imagePath0 = 'C:/Users/Cauli/Desktop/44/oneone.png'
imagePath1 = 'C:/Users/Cauli/Desktop/44/twotwo.png'
imagePath2 = 'C:/Users/Cauli/Desktop/44/out.png'
imagePath3 = 'C:/Users/Cauli/Desktop/44/frame10i11_RGB.png'
img11=Image.open(imagePath0)
img22=Image.open(imagePath1)
imgout=Image.open(imagePath2)
imgtruth=Image.open(imagePath3)
img11=np.array(img11)
img22=np.array(img22)
imgout = np.array(imgout)
imgtruth = np.array(imgtruth)
diff11 = img11.astype(np.int32)-imgout.astype(np.int32)
diff22 = img22.astype(np.int32)-imgout.astype(np.int32)
add=abs(img11.astype(np.int32)-img22.astype(np.int32))
mins_outtruth=abs(imgtruth.astype(np.int32)-imgout.astype(np.int32))
new_mins_outtruth=mins_outtruth.transpose((2,0,1))
mins=img11.astype(np.int32)-img22.astype(np.int32)
addpic=Image.fromarray(np.uint8(add))
minspic=Image.fromarray(np.uint8(mins))
mins_outtruth_pic=Image.fromarray(np.uint8(mins_outtruth))
addpic.save("C:/Users/Cauli/Desktop/44/addpic.png")
minspic.save("C:/Users/Cauli/Desktop/44/minspic.png")
mins_outtruth_pic.save("C:/Users/Cauli/Desktop/44/mins_outtruth_pic.png")
im_diff11 = Image.fromarray(np.uint8(diff11))
im_diff22 = Image.fromarray(np.uint8(diff22))
im_diff11.save("C:/Users/Cauli/Desktop/44/diff11.png")
im_diff22.save("C:/Users/Cauli/Desktop/44/diff22.png")
print(0)
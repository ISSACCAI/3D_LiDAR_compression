import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from rangeimg import LaserScan


def pre_process(o_image, n_image):
    for i in range(o_image.shape[0]):
        for j in range(o_image.shape[1]):
            if o_image[i, j] == 0:
                if i == 0:
                    if j == 0:
                        if o_image[i + 1, j] != 0 and o_image[i, j + 1] != 0:
                            n_image[i, j] = (o_image[i + 1, j] + o_image[i, j + 1]) / 2
                        else:
                            n_image[i, j] = o_image[i, j]
                    elif j == o_image.shape[1] - 1:
                        if o_image[i + 1, j] != 0 and o_image[i, j - 1] != 0:
                            n_image[i, j] = (o_image[i + 1, j] + o_image[i, j - 1]) / 2
                        else:
                            n_image[i, j] = o_image[i, j]
                    else:
                        if o_image[i + 1, j] != 0 and o_image[i, j + 1] != 0 and o_image[i, j - 1] != 0:
                            n_image[i, j] = (o_image[i + 1, j] + o_image[i, j + 1] + o_image[i, j - 1]) / 3
                        else:
                            n_image[i, j] = o_image[i, j]
                elif i == o_image.shape[0]:
                    if j == 0:
                        if o_image[i - 1, j] != 0 and o_image[i, j + 1] != 0:
                            n_image[i, j] = (o_image[i - 1, j] + o_image[i, j + 1]) / 2
                        else:
                            n_image[i, j] = o_image[i, j]
                    elif j == o_image.shape[1] - 1:
                        if o_image[i - 1, j] != 0 and o_image[i, j - 1] != 0:
                            n_image[i, j] = (o_image[i - 1, j] + o_image[i, j - 1]) / 2
                        else:
                            n_image[i, j] = o_image[i, j]
                    else:
                        if o_image[i - 1, j] != 0 and o_image[i, j + 1] != 0 and o_image[i, j - 1] != 0:
                            n_image[i, j] = (o_image[i - 1, j] + o_image[i, j + 1] + o_image[i, j - 1]) / 3
                        else:
                            n_image[i, j] = o_image[i, j]
                else:
                    if j == 0:
                        if o_image[i - 1, j] != 0 and o_image[i + 1, j] != 0 and o_image[i, j + 1] != 0:
                            n_image[i, j] = (o_image[i - 1, j] + o_image[i, j + 1] + o_image[i + 1, j]) / 3
                        else:
                            n_image[i, j] = o_image[i, j]
                    elif j == o_image.shape[1] - 1:
                        if o_image[i - 1, j] != 0 and o_image[i + 1, j] != 0 and o_image[i, j - 1] != 0:
                            n_image[i, j] = (o_image[i - 1, j] + o_image[i, j - 1] + o_image[i + 1, j]) / 3
                        else:
                            n_image[i, j] = o_image[i, j]
                    else:
                        if o_image[i - 1, j] != 0 and o_image[i, j + 1] != 0 and o_image[i, j - 1] != 0 and o_image[i + 1, j] != 0:
                            n_image[i, j] = (o_image[i - 1, j] + o_image[i, j + 1] + o_image[i, j - 1] + o_image[i + 1, j]) / 4
                        else:
                            n_image[i, j] = o_image[i, j]
            else:
                n_image[i, j] = o_image[i, j]
    return n_image





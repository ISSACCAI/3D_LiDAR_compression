import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


class DBreader_frame_interpolation(Dataset):

    def __init__(self, db_dir, resize=None, randomresizedcrop=None):
        if (resize is not None) or (randomresizedcrop is not None):
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomResizedCrop(randomresizedcrop),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(0.2),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])

        self.triplet_list = np.array([(db_dir + '/' + f) for f in listdir(db_dir) if isdir(join(db_dir, f))])  # 训练集中三元组的数目
        self.file_len = len(self.triplet_list)

    def __getitem__(self, index):
        frame0 = self.transform(Image.open(self.triplet_list[index] + "/frame0.png"))
        frame1 = self.transform(Image.open(self.triplet_list[index] + "/frame1.png"))
        frame2 = self.transform(Image.open(self.triplet_list[index] + "/frame2.png"))
        return frame0, frame1, frame2

    def __len__(self):
        return self.file_len

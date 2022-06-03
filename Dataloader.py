import numpy as np
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import os
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, imgs_dir):
        # self.imgs_path = "/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/dummy_dataset/TrainDir/NonDefect/"
        self.imgs_dir = imgs_dir
        self.file_list = os.listdir(self.imgs_dir)

        self.len = len(self.file_list)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = cv2.imread( self.imgs_dir + img_path)
        if img is None:
            print('Wrong path:')
            exit(-1)
        else:
            img = cv2.resize(img, dsize=(512, 512))
            # img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)

        img_tensor = self.transform(img)
        # img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor, img_tensor


if __name__ == '__main__':
    dataset = CustomDataset()
    print(dataset.__len__())
    print(dataset.__getitem__(15000))
    data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    for imgs, labels in data_loader:
        print("Batch of images has shape: ", imgs.shape)
        print("Batch of images has labels: ", labels.shape)

import torch
from torchvision.transforms import Compose, ToTensor
from PIL import Image
import cv2
import os
from model import AnomalyAE

# model = AnomalyAE()
# model.load_state_dict(torch.load('tensorboard_logs_03052022_13-49(for dummy)/models/best_model_25_loss=-0.0001.pt'))
# model.eval()
# model.to('cuda')

path = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/dummy_dataset/ValDir/NonDefect/'
for i in os.listdir(path):
    if '.png' in i:
        img = cv2.imread(path + i)
        transform = Compose([ToTensor()])
        img = transform(img)



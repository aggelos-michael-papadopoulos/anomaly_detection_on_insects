from model import AnomalyAE
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np
import os

model = AnomalyAE()
model.load_state_dict(torch.load('tensorboard_logs_25052022_17-02/models/best_model_17_loss=-0.0012.pt'))
model.eval()
model = model.to('cuda')

anomaly = r'EntocycleEntocycle_13DOL_458frame.png'

img = cv2.imread(anomaly)
img = cv2.resize(img, (512, 512))

transform = Compose([ToTensor()])
img = transform(img)
img = img.to('cuda')
print(img.shape)
img = img.unsqueeze(0)
y = model(img)
y = torch.abs(img[0][0] - y[0][0])
# print(y.shape)
img = img.detach().cpu().numpy()[0][0]

plt.figure(figsize=(15, 10))
plt.subplot(121)
plt.imshow(img)
plt.title('Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(y.detach().cpu().numpy() > 0.12)                            # 0.015
plt.title('Residual Thresholded')
plt.axis('off')
plt.savefig('sample_detection.png', bbox_inches='tight')
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
model.load_state_dict(torch.load('Entocycle: tensorboard_logs_05052022_11-39/models/best_model_2_loss=-0.0001.pt'))
model.eval()
model = model.to('cuda')

anomaly = r'EntocycleEntocycle_13DOL_458frame.png'

# img = Image.open(anomaly).convert('RGB')
img = cv2.imread(anomaly)
img = cv2.resize(img, (1200, 1200))
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          ???
transform = Compose([ToTensor()])
img = transform(img)
img = img.to('cuda')
print(img.shape)
img = img.unsqueeze(0)
y = model(img)
residual = torch.abs(img[0][0] - y[0][0])


THRESHOLD = 0.17    # for ento2

residual_image = residual.detach().cpu().numpy()
img = img.detach().cpu().numpy()[0][0]
plt.imshow(residual_image > THRESHOLD)                            # 0.015
plt.title('image to detect')
plt.axis('off')
plt.savefig('image to detect.png', bbox_inches='tight')


image = cv2.imread('image to detect.png')


image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([30, 30, 30], dtype="uint8")
upper = np.array([90, 255, 255], dtype="uint8")

# threshold color
thresh_color = cv2.inRange(image, lower, upper)

average = cv2.mean(thresh_color)[0]
defection = True
# print("average =", average)
if average == 0:
    detect_defection = not defection
    print(detect_defection)
else:
    detect_defection = defection
    print(detect_defection)

# write thresholded image to disk
cv2.imwrite("defection.png", thresh_color)


plt.figure(figsize=(15, 10))
plt.subplot(121)
plt.imshow(img)
plt.title('Image')
plt.axis('off')
plt.subplot(122)
plt.imshow(residual.detach().cpu().numpy() > THRESHOLD)                            # 0.015
plt.title('Residual Thresholded')
plt.axis('off')
plt.savefig('sample_detection.png', bbox_inches='tight')



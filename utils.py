import torch.nn as nn
import torch.nn.functional as F
import torch
import json
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
import os


def writeJson(data,path):
    with open(path,'w') as fp:
        json.dump(data,fp,indent=1)

class AnomalyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, (11, 11), stride=(1, 1), padding=5)
        self.bn1 = nn.BatchNorm2d(48)

        self.conv2 = nn.Conv2d(48, 48, (9, 9), stride=(2, 2), padding=4)
        self.bn2 = nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(48, 48, (7, 7), stride=(2, 2), padding=3)
        self.bn3 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48, 48, (5, 5), stride=(2, 2), padding=2)
        self.bn4 = nn.BatchNorm2d(48)

        self.conv5 = nn.Conv2d(48, 48, (3, 3), stride=(2, 2), padding=1)
        self.bn5 = nn.BatchNorm2d(48)

        self.conv_tr1 = nn.ConvTranspose2d(
            48, 48, (5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.bn_tr1 = nn.BatchNorm2d(48)

        self.conv_tr2 = nn.ConvTranspose2d(
            96, 48, (7, 7), stride=(2, 2), padding=3, output_padding=1)
        self.bn_tr2 = nn.BatchNorm2d(48)

        self.conv_tr3 = nn.ConvTranspose2d(
            96, 48, (9, 9), stride=(2, 2), padding=4, output_padding=1)
        self.bn_tr3 = nn.BatchNorm2d(48)

        self.conv_tr4 = nn.ConvTranspose2d(
            96, 48, (11, 11), stride=(2, 2), padding=5, output_padding=1)
        self.bn_tr4 = nn.BatchNorm2d(48)

        self.conv_output = nn.Conv2d(96, 3, (1, 1), (1, 1))
        self.bn_output = nn.BatchNorm2d(3)

    def forward(self, x):
        slope = 0.2
        x = F.leaky_relu((self.bn1(self.conv1(x))), slope)
        x1 = F.leaky_relu((self.bn2(self.conv2(x))), slope)
        x2 = F.leaky_relu((self.bn3(self.conv3(x1))), slope)
        x3 = F.leaky_relu((self.bn4(self.conv4(x2))), slope)
        x4 = F.leaky_relu((self.bn5(self.conv5(x3))), slope)

        x5 = F.leaky_relu(self.bn_tr1(self.conv_tr1(x4)), slope)
        x6 = F.leaky_relu(self.bn_tr2(
            self.conv_tr2(torch.cat([x5, x3], 1))), slope)
        x7 = F.leaky_relu(self.bn_tr3(
            self.conv_tr3(torch.cat([x6, x2], 1))), slope)
        x8 = F.leaky_relu(self.bn_tr4(
            self.conv_tr4(torch.cat([x7, x1], 1))), slope)

        output = F.leaky_relu(self.bn_output(
            self.conv_output(torch.cat([x8, x], 1))), slope)
        return output



def anomalyInit(device,weights):
    model = AnomalyAE()
    model.load_state_dict(torch.load(weights))
    model.eval()
    model = model.to(device)
    return model

def anomalyInfer(model, input, outputPath, device):
    toRet=[]
    for image in input:
        imagePath=image['path']
        imageID=image['id']
        img = cv2.imread(imagePath)
        img = cv2.resize(img, (1200, 1200))
        transform = Compose([ToTensor()])
        img = transform(img)
        img = img.to(device)
        img = img.unsqueeze(0)
        y = model(img)
        residual = torch.abs(img[0][0] - y[0][0])

        #### ANOMALY OR NOT ####
        THRESHOLD = 0.008

        residual_image = residual.detach().cpu().numpy()
        img = img.detach().cpu().numpy()[0][0]
        plt.imshow(residual_image > THRESHOLD)
        plt.title('image to detect')
        plt.axis('off')
        plt.savefig('image to detect.png', bbox_inches='tight')

        while 1:
            try:
                image = cv2.imread('image to detect.png')
                break
            except:
                a = 1

        os.remove('image to detect.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([30, 30, 30], dtype="uint8")
        upper = np.array([90, 255, 255], dtype="uint8")

        # threshold for yellow
        thresh_color = cv2.inRange(image, lower, upper)

        average = cv2.mean(thresh_color)[0]
        anomaly = True
        if average == 0:
            anomaly_detection = not anomaly
        else:
            anomaly_detection = anomaly

        # cv2.imwrite('defection.png', thresh_color)

        #### END ANOMALY DECISION ####

        plt.figure(figsize=(15, 10))
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Image')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(residual_image > THRESHOLD)
        plt.title('Residual Thresholded')
        plt.axis('off')
        plt.savefig(outputPath+"image_"+str(imageID)+".png", bbox_inches='tight')
        temp = {'id': str(imageID), 'name':"image_"+str(imageID)+".png", "result": anomaly_detection}
        toRet.append(temp)
    writeJson(toRet, outputPath+'output.json')
    return toRet


model=anomalyInit('cpu','ICF.2: tensorboard_logs_06052022_00-10/models/best_model_11_loss=-0.0001.pt')
input=[{'path':'/home/angepapa/PycharmProjects/anomaly-detection-using-autoencoders/ICF_1DOL_32frame.png', 'id':0}]
a=anomalyInfer(model, input, '', 'cpu')
pass
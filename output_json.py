import json
import os
from labels_dictionary import *


# def get_label(image_name):
#     farm_name =







#### SOS the path of images MUST be in form "path_(numberofinsects)_.png" in order to get the label:(numberofinsects)"
import cv2

path_of_imgs = r'dataset/splitted/'

def writeJson(data,path):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=1)

to_Ret = []
id = 0
label = 2500
for img in os.listdir(path_of_imgs):
    if 'png' in img:
        path = os.path.join(path_of_imgs, img)
        print(path)
        id += 1
        temp = {'id': id, 'path':'image_'+str(path), 'label': label}
        to_Ret.append(temp)

    writeJson(to_Ret, 'dataset/splitted/' + 'output.json')


# for i in os.listdir(path_of_imgs):
#     print(i)



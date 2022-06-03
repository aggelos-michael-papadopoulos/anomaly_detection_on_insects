import image_slicer
import cv2
from PIL import Image
import numpy as np
import os
import shutil
from move_files import move_files
from img_rename import rename_files
import json
from slicer import pil2cv
import shutil
from sorted_dataset import images_in_order, img_packs                     #images organised from 0 to 3100 in our dataset


def writeJson(data,path):
    with open(path, 'w') as fp:
        json.dump(data, fp, indent=1)

#### organize my dataset of 3100 images to go from 0 to 3099 ####
original_images_datapath = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/counting_dataset/'




datapath = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/counting_dataset/tmp/'
splitted_datapath = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/counting_dataset/splitted/'


slices = ['half', 'four', 'eighth', 'sixteenth']
cut = ""
cutter = [2, 4, 8, 16]

labels = [2500, 500, 0, 0, 1500, 3000, 4000, 4000, 4000, 30000, 10000, 4000, 2000, 500, 1000, 500, 500, 500, 2000, 2500,
          2000, 2000, 1000, 2000, 0, 10, 2000, 10, 200, 150, 200]

to_Ret = []
id = 0
label_id = -1


for pack in range(len(img_packs)):
    label_id += 1
    label = labels[label_id]
    for img in img_packs[pack]:
        images = os.path.join(original_images_datapath, img)
        print(images)
        if 'png' in images:
            label = labels[label_id]
            shutil.copy(images, splitted_datapath)
            temp = {'id': id, 'path': splitted_datapath + str(img), 'label': label}
            id += 1
            to_Ret.append(temp)
            for num in range(len(cutter)):
                cut = slices[num]
                splitter = image_slicer.slice(images, cutter[num], save=False)
                image_slicer.save_tiles(splitter, directory=splitted_datapath, prefix=img.split('.')[-2] + '_'+cut)
                print(splitter)
                for i in splitter:
                    print(i)
                    filename = i.basename + '.png'
                    temp = {'id': id, 'path': splitted_datapath + filename, 'label': label//cutter[num]}
                    print(id)
                    id += 1
                    to_Ret.append(temp)
                pil2cv(datapath)

        writeJson(to_Ret, 'dataset/splitted/' + 'dataset_output.json')



# for img in os.listdir(datapath):
#     print(img)
#     if 'png' in img:
#         label = labels[label_id]
#         images = os.path.join(datapath, img)
#         shutil.copy(images, splitted_datapath)
#         temp = {'id': id, 'path': splitted_datapath + str(img), 'label': label}
#         id += 1
#         label_id += 1
#         to_Ret.append(temp)
#         for num in range(len(cutter)):
#             cut = slices[num]
#             splitter = image_slicer.slice(images, cutter[num], save=False)
#             image_slicer.save_tiles(splitter, directory=splitted_datapath, prefix=img.split('.')[-2] + '_'+cut)
#             print(splitter)
#             for i in splitter:
#                 print(i)
#                 filename = i.basename + '.png'
#                 temp = {'id': id, 'path': splitted_datapath + filename, 'label': label//cutter[num]}
#                 print(id)
#                 id += 1
#                 to_Ret.append(temp)
#             pil2cv(datapath)
#
#     writeJson(to_Ret, 'dataset/splitted/' + 'output1.json')

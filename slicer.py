import os
import image_slicer
import cv2
import numpy as np

def pil2cv(path_of_imgs):
    d = 0
    # filename = os.path.join(path, name)
    # print(filename)
    for img in os.listdir(path_of_imgs):
        if 'png' in img:
            img = cv2.imread(path_of_imgs + img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            filename = "splitted/%s.png" % (str(img))
            cv2.imwrite(filename, img)
            d += 1


def slicer(datapath, splitted_datapath):
    slices = ['half', 'four', 'eighth', 'sixteenth']
    cut = ""
    cutter = [2, 4, 8, 16]
    for num in range(len(cutter)):
        cut = slices[num]
        for img in os.listdir(datapath):
            if 'png' in img:
                images = os.path.join(datapath, img)
                splitter = image_slicer.slice(images, cutter[num], save=False)
                image_slicer.save_tiles(splitter, directory=splitted_datapath, prefix=img.split('.')[-2] + '_'+cut)
                pil2cv(datapath)
                print(splitter)


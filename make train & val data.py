import os
import shutil
import random


src_folder = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/Nasekomo/'

filenames = [file for file in os.listdir(src_folder)]
# filenames.sort()
random.seed(230)
random.shuffle(filenames)

split_1 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1]
val_filenames = filenames[split_1:]


print(len(train_filenames))
print(len(val_filenames))


train_path = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/TrainDir/'
val_path = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/ValDir/'

### run ONLY ONE time to make train and val data once ###
for img in train_filenames:
    source = src_folder + img
    destination = train_path + img

    if os.path.isfile(source):
        shutil.move(source, destination)
        print('Moved:', img)

for img in val_filenames:
    source = src_folder + img
    destination = val_path + img

    if os.path.isfile(source):
        shutil.move(source, destination)
        print('Moved:', img)

import os
import shutil
import cv2

def move_files(path_of_source_folder, path_of_destination_folder, name_to_find):
# src_folder = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/Nasekomo/TrainDir/NonDefect/'
# dst_folder = r'/home/angepapa/Desktop/papadopoulos corosect/Datasets/farm_videos_autoencoder_dataset/dataset/all data/TrainDir/NonDefect/'

    # fetch all files
    for file_name in os.listdir(path_of_source_folder):
        if name_to_find in file_name:
            # construct full file path
            source = path_of_source_folder + file_name
            destination = path_of_destination_folder + file_name
            # move only files
            if os.path.isfile(source):
                shutil.move(source, destination)
                print('Moved:', file_name)



import os


def rename_files(path_of_source_folder, how_many_pieces):
    # folder = 'where is the dataset'
    counter = 1
    for file_name in os.listdir(path_of_source_folder):
        if '.png' in file_name:
            # Construct old file name
            print(file_name)
            source = path_of_source_folder + file_name
            destination = path_of_source_folder + file_name.split('.')[-2] + '_' + how_many_pieces + '_' + str(counter) + '_' + ".png"
            os.rename(source, destination)
            counter += 1

    # print('All Files Renamed')
    #
    # print('New Names are')
    # verify the result
    res = os.listdir(path_of_source_folder)
    # print(res)


rename_files('splitted/', 'half')


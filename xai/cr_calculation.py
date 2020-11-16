import os
import zlib

image_folder_path = os.path.normpath(os.path.join(os.getcwd(), 'Images'))  # Get path to image folder
images_list = os.listdir(image_folder_path)  # get images list
cr_dict = dict()


for image in images_list:
    filename = os.path.join(image_folder_path, image)  # full path to current image

    with open(filename, "rb") as in_file:  # Compress raw data
        compressed_data = zlib.compress(in_file.read(), 9)
        unzipped_size = in_file.tell()

    with open(filename + ".cr", "wb") as compressed_file:  # Save compressed data to file
        compressed_file.write(compressed_data)
        zipped_size = compressed_file.tell()
    
    os.remove(filename + ".cr")  # remove compressed file 
    cr_dict[image] = unzipped_size / zipped_size  # calc CR and save it to dictionary

print(cr_dict)  # print dictionary with CR's
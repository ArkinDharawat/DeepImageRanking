import os

validation_path = "tiny-imagenet-200/val/"

annotation_file_path = os.path.join(validation_path, "val_annotations.txt")

annotation_file = open(annotation_file_path, "r")
for line in annotation_file:
    image, folder = line.strip().split()[0:2]

    image_path = os.path.join(validation_path, "images/"+image)
    folder_path = os.path.join(validation_path, folder)

    if not os.path.exists(folder_path):
        print(os.makedirs(folder_path))
        if not os.path.exists(os.path.join(folder_path, "images")):
            print(os.makedirs(os.path.join(folder_path, "images")))

    os.rename(image_path, os.path.join(folder_path, "images/"+image)) # TODO: maybe replace with shutil

    # break

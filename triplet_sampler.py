import glob
import json
import random
import csv
import os
import argparse
import numpy as np
import pandas as pd

class_file = open("tiny-imagenet-200/wnids.txt", "r")
classes = [x.strip() for x in class_file.readlines()]
class_file.close()

def list_pictures(directory):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files]

def get_negative_images(all_images,image_names,num_neg_images):
    random_numbers = np.arange(len(all_images))
    np.random.shuffle(random_numbers)
    if int(num_neg_images)>(len(all_images)-1):
        num_neg_images = len(all_images)-1
    neg_count = 0
    negative_images = []
    for random_number in list(random_numbers):
        if all_images[random_number] not in image_names:
            negative_images.append(all_images[random_number])
            neg_count += 1
            if neg_count>(int(num_neg_images)-1):
                break
    return negative_images

def get_positive_images(image_name,image_names,num_pos_images):
    random_numbers = np.arange(len(image_names))
    np.random.shuffle(random_numbers)
    if int(num_pos_images)>(len(image_names)-1):
        num_pos_images = len(image_names)-1
    pos_count = 0
    positive_images = []
    for random_number in list(random_numbers):
        if image_names[random_number]!= image_name:
            positive_images.append(image_names[random_number])
            pos_count += 1
            if int(pos_count)>(int(num_pos_images)-1):
                break
    return positive_images

def generate_triplets(training, dataset_path, num_neg_images,num_pos_images):
    """
    Generate query, postivie and negative image triplets for training/testing
    :param training: 0 or 1 based on training or testing dataset
    :param dataset_path: the dataset path
    :param num_neg_images: number of negative images per query
    :param num_pos_images: number of positive images per query
    :return: void
    """
    triplet_dataframe = pd.DataFrame(columns=["query", "positive", "negative"])

    all_images = []
    for class_ in classes:
        all_images +=list_pictures(os.path.join(dataset_path, class_+"/images/"))

    triplets = []
    for class_ in classes:
        image_names = list_pictures(os.path.join(dataset_path, class_+"/images/"))
        for image_name in image_names:
            image_names_set = set(image_names)
            query_image = image_name
            positive_images = get_positive_images(image_name, image_names, num_pos_images)
            for positive_image in positive_images:
                negative_images = get_negative_images(all_images, image_names_set, num_neg_images)
                for negative_image in negative_images:
                   row = {"query":query_image,
                          "positive":positive_image,
                          "negative":negative_image}
                   print(row)
                   triplet_dataframe = triplet_dataframe.append(row, ignore_index=True)
        # break

    if training:
        triplet_dataframe.to_csv("training_triplet_sample.csv", index=False)
    else:
        triplet_dataframe.to_csv("test_triplet_sample.csv", index=False)

    print(".csv generated!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--training',
                        help='A argument for training/testing')

    parser.add_argument('--num_pos_images',
                        help='A argument for the number of Positive images per Query image')

    parser.add_argument('--num_neg_images',
                        help='A argument for the number of Negative images per Query image')

    args = parser.parse_args()

    if int(args.training) not in [0, 1]:
        print('This should be a 0 or 1 value!')
        quit()
    elif int(args.num_neg_images) < 1:
        print('Number of Negative Images cannot be less than 1!')
    elif int(args.num_pos_images) < 1:
        print('Number of Positive Images cannot be less than 1!')

    if int(args.training):
        dataset_path = "tiny-imagenet-200/train/"
    else:
        dataset_path = "tiny-imagenet-200/val/"

    if not os.path.exists(dataset_path):
        print(dataset_path + " path does not exist!")
        quit()

    print("Are we training? " + args.training)
    print("Grabbing images from: " + dataset_path)
    print("Number of Positive image per Query image: " + args.num_pos_images)
    print("Number of Negative image per Query image: " + args.num_neg_images)


    generate_triplets(int(args.training), dataset_path, args.num_neg_images, args.num_pos_images)


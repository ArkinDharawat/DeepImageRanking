#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:51:07 2020

@author: khan
"""
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

import os
from PIL import Image
from sklearn.neighbors import NearestNeighbors

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

transform_test = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform(images):
    dataset = []
    for image in images:
        img = Image.open(image).convert('RGB')
        dataset.append(transform_test(img))
    return dataset

def load_train_images(triplet_file):
    # list of traning images names, e.g., "../tiny-imagenet-200/train/n01629819/images/n01629819_238.JPEG"
    training_images = []
    for line in open(triplet_file):
        line_array = line.split(",")
        if line_array[0] not in training_images:
            training_images.append(line_array[0])
            
    return training_images 
    
def gen_test_embedding(transformed_imgs, model, pick_one):
    
    img = transformed_imgs[pick_one].reshape(1, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        test_img_embed = model(img.to(device))
    return test_img_embed
 
def NN_model(X, K=500):
    # n_neighbors is 500 because there are 200 classes and images are 100,000
    neighbor_model = NearestNeighbors(n_neighbors=500, algorithm='kd_tree', n_jobs=-1)
    neighbor_model.fit(X)
    return neighbor_model

def show_results(results, train_imgs_p, query):
    plt.subplot(1, 6, 1)
    plt.imshow(np.asarray(Image.open(query)))
    
    for indx, val in enumerate(results[1][0]):
        if indx == 5:
            break
        else:
            plt.subplot(1, 6, indx+2)
            a = np.asarray(Image.open(train_imgs_p[val]))
            plt.axis('off')
            plt.imshow(a)
    
    plt.show()
 
  
train_imgs_p = load_train_images("../triplets.txt")
train_embeddings = np.fromfile("train_embedding.txt", dtype=np.float32).reshape(-1, 4096)

neighbor_model = NN_model(train_embeddings)

test_folder = "tiny-imagenet-200/test/images"
test_imgs_p = [os.path.join(test_folder, i) for i in os.listdir(test_folder)]
test_transformed = transform(test_imgs_p)

model = torch.load('../deepranknet.model').to(device)

while True:
    choice = input("Input the Index of Query Image?(or to quit press any other key) ")
    try:
        pick_one = int(choice)
    except:
        break
    
    test_img_embedding = gen_test_embedding(test_transformed, model, pick_one)

    predictions = neighbor_model.kneighbors(test_img_embedding.cpu().detach().numpy())

    show_results(predictions, train_imgs_p, test_imgs_p[pick_one])
    
    
    
    
    
    



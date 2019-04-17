import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
#testing
import numpy as np
import time

from DeepRankNet import DeepRank

from DatasetLoader import DatasetImageNet

use_cuda = torch.cuda.is_available()
print("Cuda?: "+str(use_cuda))

BATCH_SIZE = 4
transform_train = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = DatasetImageNet("training_triplet_sample.csv", transform=transform_train)
trainloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

model = DeepRank()
checkpoint = torch.load("mytraining.pt")
model.load_state_dict(checkpoint['state_dict'])

embedded_features = []
for batch_idx, (X_train_query, X_train_postive, X_train_negative) in enumerate(trainloader):

            if (X_train_query.shape[0] < BATCH_SIZE):
                continue

            if use_cuda:
                X_train_query = Variable(X_train_query).cuda()
                X_train_postive = Variable(X_train_postive).cuda()
                X_train_negative = Variable(X_train_negative).cuda()
            else:
                X_train_query = Variable(X_train_query)#.cuda()
                X_train_postive = Variable(X_train_postive)#.cuda()
                X_train_negative = Variable(X_train_negative)#.cuda()


            X_embedding = model(X_train_query)





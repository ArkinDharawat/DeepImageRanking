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

# from DeepRankNet import DeepRank

from DatasetLoader import DatasetImageNet

use_cuda = torch.cuda.is_available()
print("Cuda?: "+str(use_cuda))


def correct_triplet(anchor, positive, negative, size_average=False):
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + 1.0)
    return losses.mean() if size_average else losses.sum()

BATCH_SIZE = 24
transform_train = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = DatasetImageNet("training_triplet_sample.csv", transform=transform_train)
trainloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)


test_dataset = DatasetImageNet("test_triplet_sample.csv", transform=transform_train)
testloader =  torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)



# model = DeepRank()
model = torch.load('deepranknet.model')
for param in model.parameters():
    param.requires_grad = False
if use_cuda:
    model.cuda()
model.eval()


print("Generating train embedding...")

embedded_features = []
triplet_ranks = 0
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


            embedding = model(X_train_query)
            embedding_p = model(X_train_postive)
            embedding_n = model(X_train_negative)
            embedding_np = embedding.cpu().detach().numpy()



            embedded_features.append(embedding_np)

            correctly_ranked_triplets = correct_triplet(embedding, embedding_p, embedding_n)
            triplet_ranks += correctly_ranked_triplets

            # break

print("Train triplets ranked correctly:", triplet_ranks, triplet_ranks/100000.)
embedded_features_train = np.concatenate(embedded_features, axis=0)

embedded_features_train.astype('float32').tofile('train_embedding.txt') # save trained embedding

embedded_features = []
triplet_ranks = 0
print("Generating test embedding...")
for batch_idx, (X_test_query, X_test_positive, X_test_negative) in enumerate(testloader):

    if (X_test_query.shape[0] < BATCH_SIZE):
        continue

    if use_cuda:
        X_test_query = Variable(X_test_query).cuda()
        X_test_positive = Variable(X_test_positive).cuda()
        X_test_negative = Variable(X_test_negative).cuda()
    else:
        X_test_query = Variable(X_test_query)  # .cuda()
        X_test_positive = Variable(X_test_positive)# .cuda()
        X_test_negative = Variable(X_test_negative)# .cuda()


    embedding = model(X_test_query)
    embedding_p = model(X_test_positive)
    embedding_n = model(X_test_negative)

    embedding_np = embedding.cpu().detach().numpy()

    embedded_features.append(embedding_np)

    correctly_ranked_triplets = correct_triplet(embedding, embedding_p, embedding_n)
    triplet_ranks += correctly_ranked_triplets

print("Test triplets ranked correctly:", triplet_ranks, triplet_ranks/2000.)

embedded_features_test = np.concatenate(embedded_features, axis=0)

embedded_features_test.astype('float32').tofile('test_embedding.txt') # save trained embedding

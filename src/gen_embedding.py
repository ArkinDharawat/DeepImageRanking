# valing
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from DeepImageRanking.src.datasetloader import DatasetImageNet

use_cuda = torch.cuda.is_available()
print("Cuda?: " + str(use_cuda))


def correct_triplet(anchor, positive, negative, size_average=False):
    """calculate triplet hinge loss

    Parameters
    ----------
    anchor
    positive
    negative
    size_average

    Returns
    -------
    float, denotes number of incorrect triplets
    """
    distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
    distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(distance_positive - distance_negative + 1.0)
    losses = (losses > 0)
    # print(losses)
    return losses.mean() if size_average else losses.sum()


BATCH_SIZE = 25
transform_train = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = DatasetImageNet("../training_triplet_sample.csv", transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

val_dataset = DatasetImageNet("../val_triplet_sample.csv", transform=transform_val)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

# model = DeepRank()
model = torch.load('../deepranknet.model')
for param in model.parameters():
    param.requires_grad = False
if use_cuda:
    model.cuda()
model.eval()

print("Generating train embedding...")
embedded_features = []
triplet_ranks = 0
batches = 0
for batch_idx, (X_train_query, X_train_postive, X_train_negative) in enumerate(trainloader):
    if (X_train_query.shape[0] < BATCH_SIZE):
        continue

    if use_cuda:
        X_train_query = Variable(X_train_query).cuda()
        X_train_postive = Variable(X_train_postive).cuda()
        X_train_negative = Variable(X_train_negative).cuda()
    else:
        X_train_query = Variable(X_train_query)
        X_train_postive = Variable(X_train_postive)
        X_train_negative = Variable(X_train_negative)

    batches += 1
    embedding = model(X_train_query)
    embedding_p = model(X_train_postive)
    embedding_n = model(X_train_negative)
    embedding_np = embedding.cpu().detach().numpy()

    embedded_features.append(embedding_np)

    incorrectly_ranked_triplets = correct_triplet(embedding, embedding_p, embedding_n)
    triplet_ranks += incorrectly_ranked_triplets

    # break

print("Train triplets ranked correctly:", (batches * BATCH_SIZE) - triplet_ranks,
      1 - float(triplet_ranks) / (batches * BATCH_SIZE))
embedded_features_train = np.concatenate(embedded_features, axis=0)

embedded_features_train.astype('float32').tofile('../train_embedding.txt')  # save trained embedding

embedded_features = []
triplet_ranks = 0
print("Generating validation data embedding...")
batches = 0
for batch_idx, (X_val_query, X_val_positive, X_val_negative) in enumerate(valloader):

    if (X_val_query.shape[0] < BATCH_SIZE):
        continue

    if use_cuda:
        X_val_query = Variable(X_val_query).cuda()
        X_val_positive = Variable(X_val_positive).cuda()
        X_val_negative = Variable(X_val_negative).cuda()
    else:
        X_val_query = Variable(X_val_query)  # .cuda()
        X_val_positive = Variable(X_val_positive)  # .cuda()
        X_val_negative = Variable(X_val_negative)  # .cuda()

    batches += 1
    embedding = model(X_val_query)
    embedding_p = model(X_val_positive)
    embedding_n = model(X_val_negative)

    embedding_np = embedding.cpu().detach().numpy()

    embedded_features.append(embedding_np)

    incorrectly_ranked_triplets = correct_triplet(embedding, embedding_p, embedding_n)
    triplet_ranks += incorrectly_ranked_triplets

print("validation triplets ranked correctly:", (batches * BATCH_SIZE) - triplet_ranks,
      1 - float(triplet_ranks) / (batches * BATCH_SIZE))

embedded_features_val = np.concatenate(embedded_features, axis=0)

embedded_features_val.astype('float32').tofile('../val_embedding.txt')  # save trained embedding

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from DeepImageRanking.src.datasetloader import DatasetImageNet
from DeepImageRanking.src.deep_rank_net import DeepRank

BATCH_SIZE = 25
LEARNING_RATE = 0.001
VERBOSE = 1

if VERBOSE:
    print("Libs locked and loaded")

use_cuda = torch.cuda.is_available()
print("Cuda?: " + str(use_cuda))

transform_train = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# TODO: add transformer for dataset?
train_dataset = DatasetImageNet("../training_triplet_sample.csv", transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32)


def train_and_eval_model(num_epochs, optim_name=""):
    model = DeepRank()

    sub_model = list(model.conv_model.features.children())[:-2]
    for s in sub_model:
        for param in s.parameters():
            param.requires_grad = False  # switch off all gradients except last two

    if use_cuda:
        model.cuda()

    if optim_name == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    elif optim_name == "rms":
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, momentum=0.9,
                              nesterov=True)

    model.train()
    train_loss = []
    mean_train_loss = []

    start_time = time.time()

    for epoch in range(0, num_epochs):
        if VERBOSE:
            print("Epoch is: " + str(epoch))

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

            optimizer.zero_grad()  # set gradient to 0

            query_embedding = model(X_train_query)
            postive_embedding = model(X_train_postive)
            negative_embedding = model(X_train_negative)

            loss = F.triplet_margin_loss(anchor=query_embedding, positive=postive_embedding,
                                         negative=negative_embedding)

            loss.backward()

            train_loss.append(loss.data[0])
            optimizer.step()

        torch.save(model, 'temp_net' + str(epoch + 1) + '.model')  # temporary model to save
        loss_epoch = np.mean(train_loss)
        mean_train_loss.append(loss_epoch)
        print(epoch, loss_epoch)

    end_time = time.time()

    np.asarray(mean_train_loss).astype('float32').tofile('mean_training_loss.txt')

    print("Total training time " + str(end_time - start_time))

    torch.save(model, '../deepranknet.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('--epochs',
                        help='A argument for no. of epochs')

    parser.add_argument('--optim',
                        help='A argument for the optimizer to choose eg: adam')

    args = parser.parse_args()
    epochs = 0
    if int(args.epochs) < 0:
        print('This should be a positive value')
        quit()
    else:
        epochs = int(args.epochs)

    if str(args.optim) not in ["adam", "rms"]:
        print('switching to default optimizer, SGD+Momentum')
        optim_name = "sgd"
    else:
        optim_name = args.optim.lower()

    train_and_eval_model(num_epochs=epochs, optim_name=optim_name)

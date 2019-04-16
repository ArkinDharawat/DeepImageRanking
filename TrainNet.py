import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
#testing
import time

from DeepRankNet import DeepRank

from DatasetLoader import DatasetImageNet

DIM = 32
BATCH_SIZE = 4
LEARNING_RATE =  0.001
VERBOSE  = 1

if VERBOSE:
    print("Libs locked and loaded")

use_cuda = torch.cuda.is_available()
print("Cuda?: "+str(use_cuda))


transform_train = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





#TODO: add transformer for dataset?
train_dataset = DatasetImageNet("training_triplet_sample.csv", transform=transform_train)
trainloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

test_dataset = DatasetImageNet("test_triplet_sample.csv", transform=transform_train)
testloader =  torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


def train_and_eval_model(num_epochs, optim_name=""):
    model = DeepRank()

    if use_cuda:
        model.cuda()

    if optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optim_name == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)

    model.train()
    train_loss = []

    criterion = nn.TripletMarginLoss(margin=1, p=2)

    start_time = time.time()

    for epoch in range(0, num_epochs):
        train_accu = []
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
                X_train_query = Variable(X_train_query)#.cuda()
                X_train_postive = Variable(X_train_postive)#.cuda()
                X_train_negative = Variable(X_train_negative)#.cuda()

            optimizer.zero_grad()  # set gradient to 0

            query_embedding = model(X_train_query)
            postive_embedding = model(X_train_postive)
            negative_embedding = model(X_train_negative)

            loss = F.triplet_margin_loss(anchor=query_embedding, positive=postive_embedding, negative=negative_embedding)

            print(torch.norm(loss, p=2))

            loss.backward()
            # break
        break
        accuracy_epoch = np.mean(train_accu)
        loss_epoch = np.mean(train_loss)
        print(epoch, accuracy_epoch, loss_epoch)

    end_time = time.time()
    print("Total training time " + str(end_time - start_time))

if __name__ == '__main__':
    train_and_eval_model(num_epochs=1)

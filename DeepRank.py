import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
#testing
import time

from DatasetLoader import DatasetImageNet

DIM = 32
BATCH_SIZE = 4
LEARNING_RATE =  0.001
VERBOSE  = 1

if VERBOSE:
    print("Libs locked and loaded")

use_cuda = torch.cuda.is_available()
print("Cuda?: "+str(use_cuda))

class BasicBlock(nn.Module):
    """
    Basicblock for ResNet101 arch
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.downsample_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        save_x = x

        output_bb = self.conv1(x)
        output_bb = F.relu(self.batchnorm1(output_bb))
        output_bb = self.conv2(output_bb)
        output_bb = self.batchnorm2(output_bb)

        if self.downsample:
            save_x = self.downsample_layer(save_x)

        output_bb += save_x
        output_bb = F.relu(output_bb)

        return output_bb


class ResNet(nn.Module):
    def __init__(self):
        """
        ResNet101 Arch
        """
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)

        # 2
        out_channels_2 = 32
        self.basicblock_20 = BasicBlock(in_channels=32, out_channels=out_channels_2)
        self.basicblock_21 = BasicBlock(in_channels=out_channels_2, out_channels=out_channels_2)

        # 3
        out_channels_3 = 64
        self.basicblock_30 = BasicBlock(in_channels=out_channels_2, out_channels=out_channels_3, stride=2, downsample=True)
        self.basicblock_31 = BasicBlock(in_channels=out_channels_3, out_channels=out_channels_3)
        self.basicblock_32 = BasicBlock(in_channels=out_channels_3, out_channels=out_channels_3)
        self.basicblock_33 = BasicBlock(in_channels=out_channels_3, out_channels=out_channels_3)

        # 4
        out_channels_4 = 128
        self.basicblock_40 = BasicBlock(in_channels=out_channels_3, out_channels=out_channels_4, stride=2, downsample=True)
        self.basicblock_41 = BasicBlock(in_channels=out_channels_4, out_channels=out_channels_4)
        self.basicblock_42 = BasicBlock(in_channels=out_channels_4, out_channels=out_channels_4)
        self.basicblock_43 = BasicBlock(in_channels=out_channels_4, out_channels=out_channels_4)

        # 5
        out_channels_5 = 256
        self.basicblock_50 = BasicBlock(in_channels=out_channels_4, out_channels=out_channels_5, stride=2, downsample=True)
        self.basicblock_51 = BasicBlock(in_channels=out_channels_5, out_channels=out_channels_5)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) #


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.batchnorm1(x))
        x = F.dropout(x, p=0.6)

        x = self.basicblock_20(x)
        x = self.basicblock_21(x)

        x = self.basicblock_30(x)
        x = self.basicblock_31(x)
        x = self.basicblock_32(x)
        x = self.basicblock_33(x)

        x = self.basicblock_40(x)
        x = self.basicblock_41(x)
        x = self.basicblock_42(x)
        x = self.basicblock_43(x)

        x = self.basicblock_50(x)
        x = self.basicblock_51(x)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)

        return x

class DeepRank(nn.Module):
    """
    Deep Image Rank Architecture
    """
    def __init__(self):
        super(DeepRank, self).__init__()

        self.conv_model = ResNet()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1, stride=4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=4, padding=0)

        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1, stride=8)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dense_layer = torch.nn.Linear(in_features=(4096+3072), out_features=4096)


    def forward(self, X):
        conv_input = self.conv_model(X)
        conv_norm = conv_input.norm(p=2, dim=1, keepdim=True)
        conv_input = conv_input.div(conv_norm.expand_as(conv_input))

        first_input = self.conv1(X)
        first_input = self.maxpool1(first_input)
        first_input = first_input.view(first_input.size(0), -1)
        first_norm = first_input.norm(p=2, dim=1, keepdim=True)
        first_input = first_input.div(first_norm.expand_as(first_input))


        second_input = self.conv2(X)
        second_input = self.maxpool2(second_input)
        second_input = second_input.view(second_input.size(0), -1)
        second_norm = second_input.norm(p=2, dim=1, keepdim=True)
        second_input = second_input.div(second_norm.expand_as(second_input))

        merge_subsample = torch.cat([first_input, second_input], 1) # batch x (3076)

        merge_conv = torch.cat([merge_subsample, conv_input], 1) #  batch x (4096 + 3076)

        final_input = self.dense_layer(merge_conv)
        final_norm = final_input.norm(p=2, dim=1, keepdim=True)
        final_input = final_input.div(final_norm.expand_as(final_input))

        return final_input

def custom_loss(y_pred):
    """
    Custom Loss function for Image ranking
    :param y_pred:
    :return:


    """
    g = torch.Tensor(1)
    loss = Variable(torch.Tensor(0.)).cuda()

    for i in range(0, BATCH_SIZE):
        q_embedding = y_pred[i + 0]
        p_embedding = y_pred[i + BATCH_SIZE]
        n_embedding = y_pred[i + BATCH_SIZE*2]

        D_q_p = torch.pow(q_embedding + p_embedding, 0.5)
        D_q_n = torch.pow(q_embedding + n_embedding, 0.5)
        loss = (loss + g + D_q_p - D_q_n)

    loss = loss/(BATCH_SIZE)
    zero =  Variable(torch.Tensor(0.)).cuda()

    return torch.max(zero, loss)



#TODO: add transformer for dataset?
train_dataset = DatasetImageNet("training_triplet_sample.csv", None)
trainloader =  torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

test_dataset = DatasetImageNet("test_triplet_sample.csv", None)
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

            image_embedding = torch.cat([query_embedding, postive_embedding, negative_embedding], 0)

            loss = custom_loss(image_embedding) #TODO: REVIEW THIS AGAIN

            break
        break

    end_time = time.time()
    print("Total training time " + str(end_time - start_time))

if __name__ == '__main__':
    train_and_eval_model(num_epochs=1)

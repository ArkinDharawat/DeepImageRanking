import os
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable


# def resnet101(pretrained=False, **kwargs):
#     """
#     Construct a ResNet-101 model.
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = torchvision.models.resnet.ResNet(
#         torchvision.models.resnet.BasicBlock, [3, 4, 23, 3])
#     if pretrained:
#         model.load_state_dict(torch.utils.model_zoo.load_url(
#             'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', model_dir='../resnet101'), strict=False)
#     return ConvNet(model)


class ConvNet(nn.Module):
    """EmbeddingNet using ResNet-101."""

    def __init__(self):
        """Initialize EmbeddingNet model."""
        super(ConvNet, self).__init__()

        # Everything except the last linear layer
        resnet = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_ftrs = resnet.fc.in_features
        self.fc1 = nn.Linear(num_ftrs, 4096)

    def forward(self, x):
        """Forward pass of EmbeddingNet."""

        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)

        return out

class DeepRank(nn.Module):
    """
    Deep Image Rank Architecture
    """
    def __init__(self):
        super(DeepRank, self).__init__()

        self.conv_model = ConvNet()

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=1, stride=16)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=4, padding=1)

        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=8, padding=4, stride=32)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=3)

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

        merge_subsample = torch.cat([first_input, second_input], 1) # batch x (3072)

        merge_conv = torch.cat([merge_subsample, conv_input], 1) #  batch x (4096 + 3072)


        final_input = self.dense_layer(merge_conv)
        final_norm = final_input.norm(p=2, dim=1, keepdim=True)
        final_input = final_input.div(final_norm.expand_as(final_input))

        return final_input

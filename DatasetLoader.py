import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.image import imread

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from PIL import Image

class DatasetImageNet(Dataset):
    """
    Dataset class for tiny Image net dataset
    """
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index,:]
        images = [Image.open(row[i]).convert('RGB') for i in range(0, 3)]

        if self.transform is not None:
            for i in range(0, 3):
                images[i] = self.transform(images[i])

        q_image, p_image, n_image = images[0], images[1], images[2]

        return q_image, p_image, n_image


def test_dataloader():
    """
    Method to test dataloader
    :return: Void
    """
    transform_train = transforms.Compose([
        transforms.Resize(224, interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DatasetImageNet("training_triplet_sample.csv", transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    train_iter = iter(train_loader)
    print(type(train_iter))

    q,p,n = train_iter.next()

    print('images shape on batch size = {}'.format(q.size()))
    print('images shape on batch size = {}'.format(p.size()))
    print('images shape on batch size = {}'.format(n.size()))

if __name__ == '__main__':
    test_dataloader()
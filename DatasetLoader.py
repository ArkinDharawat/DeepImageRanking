import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.image import imread

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

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
        images = [imread(row[i]).astype(np.float32) for i in range(0, 3)]

        for i in range(0, 3):
            if images[i].shape == (64, 64):
                images[i] = np.asarray([images[i][:], images[i][:], images[i][:]]) # TODO: CHANE RGB TO GRAYSCALE
            images[i] = images[i][:].reshape((3, 64, 64))

        q_image, p_image, n_image = images[0], images[1], images[2]

        if self.transform is not None:
            q_image = self.transform(q_image) # TODO: Apply to all or only to query?

        return q_image, p_image, n_image


def test_dataloader():
    """
    Method to test dataloader
    :return: Void
    """
    train_dataset = DatasetImageNet("training_triplet_sample.csv", None)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    train_iter = iter(train_loader)
    print(type(train_iter))

    q,p,n = train_iter.next()

    print('images shape on batch size = {}'.format(q.size()))
    print('images shape on batch size = {}'.format(p.size()))
    print('images shape on batch size = {}'.format(n.size()))

if __name__ == '__main__':
    test_dataloader()
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __repr__(self):
        return f"Dataset(x shape={self.x.shape}, y shape={self.y.shape})"


    def __str__(self):
        return f"Dataset with {len(self.x)} samples."


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


    def __setitem__(self, idx, value):
        self.x[idx], self.y[idx] = value
        

    def __iter__(self):
        for i in range(len(self.x)):
            yield self.x[i], self.y[i]
    

    def __add__(self, other):
        if isinstance(other, Dataset):
            x_combined = np.concatenate((self.x, other.x), axis=0)
            y_combined = np.concatenate((self.y, other.y), axis=0)
            return Dataset(x_combined, y_combined)
        raise TypeError("Can only add another Dataset instance.")


def plot_sample(dataset, index):
    x = dataset.x
    y = dataset.y
    plt.figure(figsize = (15, 2))
    plt.imshow(x[index])
    plt.xlabel(CLASSES[y[index][0]])
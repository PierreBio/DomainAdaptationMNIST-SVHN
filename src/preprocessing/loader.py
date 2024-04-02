import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def dataset_exists(path):
    """Check if a dataset directory exists and has data files."""
    return os.path.exists(path) and len(os.listdir(path)) > 0

def extract_data_and_labels(train_loader, test_loader):
    """Extract data and labels from DataLoader object."""
    train_data_list = []
    train_label_list = []
    test_data_list = []
    test_label_list = []

    for data, labels in train_loader:
        train_data_list.append(data.cpu().numpy())
        train_label_list.append(labels.cpu().numpy())

    for data, labels in test_loader:
        test_data_list.append(data.cpu().numpy())
        test_label_list.append(labels.cpu().numpy())

    train_data_array = np.concatenate(train_data_list, axis=0)
    train_label_array = np.concatenate(train_label_list, axis=0)
    test_data_array = np.concatenate(test_data_list, axis=0)
    test_label_array = np.concatenate(test_label_list, axis=0)

    return SimpleDataset(train_X=train_data_array, train_y=train_label_array,
                         test_X=test_data_array, test_y=test_label_array)

def load_data():
    """
    Load MNIST and SVHN datasets.
    Apply necessary transformations/preprocessing.
    """
    mnist_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    mnist_path = './data/MNIST'
    if not dataset_exists(mnist_path):
        print("Downloading MNIST dataset...")
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    svhn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    svhn_path = './data/SVHN'
    if not dataset_exists(svhn_path):
        print("Downloading SVHN dataset...")
    svhn_train = datasets.SVHN(root='./data', split='train', download=True, transform=svhn_transform)
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=svhn_transform)

    # Data loaders
    mnist_train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True, drop_last=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False, drop_last=True)
    svhn_train_loader = DataLoader(svhn_train, batch_size=64, shuffle=True, drop_last=True)
    svhn_test_loader = DataLoader(svhn_test, batch_size=64, shuffle=False, drop_last=True)

    return mnist_train_loader, mnist_test_loader, svhn_train_loader, svhn_test_loader


class SimpleDataset:
    """Class dedicated to handle conversion from DataLoader."""
    def __init__(self, train_X=None, train_y=None, test_X=None, test_y=None):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
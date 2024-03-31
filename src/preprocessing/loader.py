import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def dataset_exists(path):
    """Check if a dataset directory exists and has data files."""
    return os.path.exists(path) and len(os.listdir(path)) > 0

def load_data():
    """
    Load MNIST and SVHN datasets.
    Apply necessary transformations/preprocessing.
    """
    # MNIST: Training and Test sets
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

    # SVHN: Training and Test sets (only images for training)
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
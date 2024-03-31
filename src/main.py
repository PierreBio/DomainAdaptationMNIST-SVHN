import torch
import torch.optim as optim
import torch.nn as nn

from src.preprocessing.loader import *
from src.model.cnn import BasicCNN
from src.model.trainer import *
from src.postprocessing.evaluator import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_train_loader, mnist_test_loader, svhn_train_loader, svhn_test_loader = load_data()
    model = BasicCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_model(model, mnist_train_loader, optimizer, criterion, device)
    # adapt_domain(model, svhn_train_loader, device)  # Implement or adjust this function based on your adaptation strategy
    evaluate_model(model, svhn_test_loader, device)

if __name__ == "__main__":
    main()
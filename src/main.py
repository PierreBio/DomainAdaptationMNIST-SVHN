import torch

from src.preprocessing.loader import *
from src.model.model_factory import *
from src.postprocessing.evaluator import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_train_loader, mnist_test_loader, svhn_train_loader, svhn_test_loader = load_data()
    model = model_factory('gan').to(device)
    model.train_model(mnist_train_loader, svhn_train_loader, device)
    # adapt_domain(model, svhn_train_loader, device)  # Implement or adjust this function based on your adaptation strategy
    evaluate_model(model, svhn_test_loader, device)

if __name__ == "__main__":
    main()
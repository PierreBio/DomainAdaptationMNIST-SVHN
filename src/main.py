import torch

from src.preprocessing.loader import load_data
from src.model.model_trainer import ModelTrainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_train_loader, mnist_test_loader, svhn_train_loader, svhn_test_loader = load_data()
    trainer = ModelTrainer(model_type='gan', device=device)
    trainer.train_and_evaluate(mnist_train_loader, svhn_train_loader, svhn_test_loader)

if __name__ == "__main__":
    main()
import torch, torch.cuda

from src.preprocessing.loader import load_data
from src.model.model_handler import ModelHandler

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mnist_train_loader, mnist_test_loader, svhn_train_loader, svhn_test_loader = load_data()
    trainer = ModelHandler(model_type='mean_teacher', device=device)
    trainer.train_and_evaluate(mnist_train_loader, svhn_train_loader, mnist_test_loader, svhn_test_loader)

if __name__ == "__main__":
    main()
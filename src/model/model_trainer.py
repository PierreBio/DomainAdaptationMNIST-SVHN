from torch.utils.data import TensorDataset, DataLoader

from src.model.model_factory import *
from src.model.gan import *
from src.postprocessing.evaluator import *

class ModelTrainer:
    def __init__(self, model_type, device):
        self.model = model_factory(model_type).to(device)
        self.device = device

    def train_and_evaluate(self, source_loader, target_loader, test_loader):
        if isinstance(self.model, GAN):
            self.model.train_model(source_loader, target_loader, self.device)
            transformed_images = self._transform_images(source_loader)
            classifier = DigitClassifier().to(self.device)
            classifier.train_model(classifier, transformed_images, self.device)
            self._evaluate_classifier(classifier, test_loader)
        elif isinstance(self.model, CNN):
            self.model.train_model(source_loader, target_loader, self.device)
            self._evaluate_classifier(self.model, test_loader)

    def _transform_images(self, source_loader):
        transformed_images = []
        labels_list = []
        self.model.generator.eval()
        with torch.no_grad():
            for images, labels in source_loader:
                images = images.to(self.device)
                transformed = self.model.generator(images).cpu()
                transformed_images.append(transformed)
                labels_list.append(labels)
        self.model.generator.train()
        transformed_dataset = TensorDataset(torch.cat(transformed_images, dim=0), torch.cat(labels_list, dim=0))
        transformed_loader = DataLoader(transformed_dataset, batch_size=64, shuffle=True)
        return transformed_loader

    def _evaluate_classifier(self, classifier, test_loader):
        evaluate_model(classifier, test_loader, self.device)

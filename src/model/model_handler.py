from src.model.model_factory import *
from src.model.gan import *
from src.model.mean_teacher import *
from src.postprocessing.evaluator import *

class ModelHandler:
    def __init__(self, model_type, device):
        self.model = model_factory(model_type).to(device)
        self.device = device

    def train_and_evaluate(self, source_loader, target_loader, source_test_loader, target_test_loader):
        if isinstance(self.model, GAN):
            self.model.train_model(source_loader, target_loader, self.device)
            transformed_images = self.model.transform_images(source_loader)
            classifier = DigitClassifier().to(self.device)
            classifier.train_model(classifier, transformed_images, self.device)
            self._evaluate_classifier(classifier, target_test_loader)
        elif isinstance(self.model, CNN):
            self.model.train_model(source_loader, target_loader, self.device)
            self._evaluate_classifier(self.model, target_test_loader)
        elif isinstance(self.model, MeanTeacher):
            classifier1, classifier2 = self.model.train_model(source_loader, target_loader, source_test_loader, target_test_loader, self.device)
            self._evaluate_classifier(classifier1, target_test_loader)
            self._evaluate_classifier(classifier2, target_test_loader)
            torch.save(classifier2.state_dict(), 'best_model.pth')

    def _evaluate_classifier(self, classifier, test_loader):
        evaluate_model(classifier, test_loader, self.device)

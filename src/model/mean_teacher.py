import torch
from torch import nn
from torch.nn import functional as F

from src.preprocessing.loader import *
from src.preprocessing.augmentation import *

class MeanTeacher(nn.Module):
    def __init__(self):
        super(MeanTeacher, self).__init__()
        self.student_net = None
        self.teacher_net = None
        self.student_optimizer = None
        self.teacher_optimizer = None
        self.classification_criterion = None
        self.device = None

    def forward(self, x):
        return x

    def train_model(self, source_loader, target_loader, source_test_loader, target_test_loader, device):
        self.device = device
        source_data = extract_data_and_labels(source_loader, source_test_loader)
        target_data = extract_data_and_labels(target_loader, target_test_loader)

        nb_classes = 10
        self.student_net = ConvolutionalClassifier(nb_classes).to(device)
        self.teacher_net = ConvolutionalClassifier(nb_classes).to(device)

        for param in self.teacher_net.parameters():
            param.requires_grad = False

        self.student_optimizer = torch.optim.Adam(self.student_net.parameters(), lr=0.001)
        self.teacher_optimizer = WeightedAverageOptimizer(self.teacher_net, self.student_net, smoothing_factor=0.01)
        self.classification_criterion = torch.nn.CrossEntropyLoss()

        best_state, best_conf_rate = {}, 0.0
        batch_size = 256
        num_epochs = 5

        for epoch in range(num_epochs):
            indices = torch.randperm(len(source_data.train_X))

            total_train_loss, total_unsup_loss, total_conf_rate, total_mask_rate = 0.0, 0.0, 0.0, 0.0
            total_samples = 0

            for i in range(0, len(source_data.train_X), batch_size):
                batch_indices = indices[i:i+batch_size]

                X_source_batch = source_data.train_X[batch_indices]
                y_source_batch = source_data.train_y[batch_indices]
                X_target_batch = target_data.train_X[batch_indices]

                X_source_aug, y_source_aug, X_target_student, X_target_teacher = self.augment_data(
                    X_source_batch, y_source_batch, X_target_batch
                )

                train_loss, unsup_loss, batch_conf_rate, batch_mask_rate  = self.train_function(
                    X_source_aug, y_source_aug, X_target_student, X_target_teacher
                )

                batch_samples = len(batch_indices)
                total_train_loss += train_loss
                total_unsup_loss += unsup_loss
                total_conf_rate += batch_conf_rate * batch_samples
                total_mask_rate += batch_mask_rate * batch_samples
                total_samples += batch_samples

                if batch_conf_rate > best_conf_rate:
                    best_conf_rate = batch_conf_rate
                    best_state = {k: v.clone().detach() for k, v in self.teacher_net.state_dict().items()}

            epoch_train_loss = total_train_loss / total_samples
            epoch_unsup_loss = total_unsup_loss / total_samples
            epoch_conf_rate = total_conf_rate / total_samples
            epoch_mask_rate = total_mask_rate / total_samples

            print(f'Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f}, Unsup Loss: {epoch_unsup_loss:.4f}, '
                  f'Conf Rate: {epoch_conf_rate:.4f}, Mask Rate: {epoch_mask_rate:.4f}')

        if best_state is not None:
            self.teacher_net.load_state_dict(best_state)
            self.teacher_net.eval()

        return self.teacher_net, self.student_net

    def augment_data(self, X_source, y_source, X_target):
        """
        Apply augmentation to source and target data.
        """
        src_aug = AugmentationSettings(2.0, -1.5, 1.5, -0.5, 0.5, 0.1)
        tgt_aug = AugmentationSettings(2.0, -1.5, 1.5, -0.5, 0.5, 0.1)

        X_source = src_aug.augment(X_source)
        X_target_student, X_target_teacher = tgt_aug.augment_pair(X_target)
        return X_source, y_source, X_target_student, X_target_teacher

    def compute_augmentation_loss(self, student_output, teacher_output, nb_classes = 10):
        """
        Compute the augmentation loss.
        """
        confidence_teacher = torch.max(teacher_output, 1)[0]
        confidence_threshold = 0.95

        unsupervised_mask = confidence_mask = (confidence_teacher > confidence_threshold).float()
        unsupervised_mask_count = confidence_mask_count = confidence_mask.sum()

        difference_aug_loss = student_output - teacher_output
        augmentation_loss = difference_aug_loss * difference_aug_loss
        augmentation_loss = augmentation_loss.mean(dim=1)
        unsupervised_loss = (augmentation_loss * unsupervised_mask).mean()
        class_balance = 0.005

        if class_balance > 0.0:
            average_class_probability = student_output.mean(dim=0)
            equalize_class_loss = self.robust_binary_crossentropy(average_class_probability, float(1.0 / nb_classes))
            equalize_class_loss = equalize_class_loss.mean() * nb_classes

            equalize_class_loss = equalize_class_loss * unsupervised_mask.mean(dim=0)
            unsupervised_loss += equalize_class_loss * class_balance

        return unsupervised_loss, confidence_mask_count, unsupervised_mask_count

    def train_function(self,  X_source, y_source, X_target_student, X_target_teacher):
        """
        Perform training of the model.
        """
        X_source = torch.tensor(X_source, dtype=torch.float, device=self.device)
        y_source = torch.tensor(y_source, dtype=torch.long, device=self.device)
        X_target_student = torch.tensor(X_target_student, dtype=torch.float, device=self.device)
        X_target_teacher = torch.tensor(X_target_teacher, dtype=torch.float, device=self.device)

        self.student_optimizer.zero_grad()
        self.student_net.train()
        self.teacher_net.train()

        src_logits_out = self.student_net(X_source)
        student_tgt_logits_out = self.student_net(X_target_student)
        student_tgt_prob_out = F.softmax(student_tgt_logits_out, dim=1)
        teacher_tgt_logits_out = self.teacher_net(X_target_teacher)
        teacher_tgt_prob_out = F.softmax(teacher_tgt_logits_out, dim=1)

        clf_loss = self.classification_criterion(src_logits_out, y_source)

        unsup_loss, conf_mask_count, unsup_mask_count = self.compute_augmentation_loss(student_tgt_prob_out, teacher_tgt_prob_out)

        unsup_weight = 3.0
        loss_expression = clf_loss + unsup_loss * unsup_weight

        loss_expression.backward()
        self.student_optimizer.step()
        self.teacher_optimizer.update_parameters()

        n_samples = X_source.size()[0]
        conf_rate = conf_mask_count / n_samples
        mask_rate = unsup_mask_count / n_samples

        return float(clf_loss) * n_samples, float(unsup_loss) * n_samples, conf_rate, mask_rate

    def predict_source_data(self, X_source):
        """
        Predicts the output probabilities for source data.
        """
        X_variable = torch.tensor(X_source, dtype=torch.float, device=self.device)
        self.student_net.eval()
        self.teacher_net.eval()
        return (F.softmax(self.student_net(X_variable), dim=1).detach().cpu().numpy(),
                F.softmax(self.teacher_net(X_variable), dim=1).detach().cpu().numpy())

    def robust_binary_crossentropy(self, pred, target):
        """
        Compute robust binary crossentropy loss.
        """
        inverted_target = -target + 1.0
        inverted_pred = -pred + 1.0 + 1e-6
        return -(target * torch.log(pred + 1.0e-6) + inverted_target * torch.log(inverted_pred))


class ConvolutionalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ConvolutionalClassifier, self).__init__()

        self.first_conv_block = nn.Sequential(
            nn.Conv2d(3, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout()
        )

        self.second_conv_block = nn.Sequential(
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout()
        )

        self.third_conv_block = nn.Sequential(
            nn.Conv2d(256, 512, (3, 3), padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, (1, 1), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, (1, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=6),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.first_conv_block(x)
        x = self.second_conv_block(x)
        x = self.third_conv_block(x)
        x = self.classifier(x)
        return x


class WeightedAverageOptimizer(object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, teacher_model, student_model, smoothing_factor=0.999):
        self.teacher_parameters = list(teacher_model.parameters())
        self.student_parameters = list(student_model.parameters())
        self.smoothing_factor = smoothing_factor
        for teacher_param, student_param in zip(self.teacher_parameters, self.student_parameters):
            teacher_param.data.copy_(student_param.data)

    def update_parameters(self):
        smoothing_complement = 1.0 - self.smoothing_factor
        for teacher_param, student_param in zip(self.teacher_parameters, self.student_parameters):
            teacher_param.data.mul_(self.smoothing_factor)
            teacher_param.data.add_(student_param.data * smoothing_complement)
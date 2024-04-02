from batchup import data_source, work_pool
import torch
from torch import nn
from torch.nn import functional as F
import time

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
        pool = work_pool.WorkerThreadPool(2)
        source_data = extract_data_and_labels(source_loader, source_test_loader)
        target_data = extract_data_and_labels(target_loader, target_test_loader)

        nb_classes = 10
        self.student_net = ConvolutionalClassifier(nb_classes).to(device)
        self.teacher_net = ConvolutionalClassifier(nb_classes).to(device)

        student_params = list(self.student_net.parameters())
        teacher_params = list(self.teacher_net.parameters())

        for param in teacher_params:
            param.requires_grad = False

        self.student_optimizer = torch.optim.Adam(student_params, lr=0.001)
        self.teacher_optimizer = WeightedAverageOptimizer(self.teacher_net, self.student_net, smoothing_factor=False)
        self.classification_criterion = nn.CrossEntropyLoss()

        print('Training Set: X shape={}, y shape={}'.format(source_data.train_X[:].shape, source_data.train_y[:].shape))
        print('Test Set: X shape={}, y shape={}'.format(source_data.test_X[:].shape, source_data.test_y[:].shape))
        print('TARGET Training Set: X shape={}'.format(target_data.train_X[:].shape))
        print('TARGET Test Set: X shape={}, y shape={}'.format(target_data.test_X[:].shape, target_data.test_y[:].shape))

        batch_sz = 256
        sup_ds = data_source.ArrayDataSource([source_data.train_X, source_data.train_y], repeats=-1)
        tgt_ds = data_source.ArrayDataSource([target_data.train_X], repeats=-1)
        train_ds = pool.parallel_data_source(data_source.CompositeDataSource([sup_ds, tgt_ds]).map(self.augment_data))
        n_batches = target_data.train_X.shape[0] // batch_sz

        source_test = data_source.ArrayDataSource([source_data.test_X, source_data.test_y])
        target_test = data_source.ArrayDataSource([target_data.test_X, target_data.test_y])

        train_iter = train_ds.batch_iterator(batch_size=batch_sz, shuffle=np.random)

        best_state, best_mask_rate = {}, 0.0
        for epoch in range(2):
            t_start = time.time()
            train_res = data_source.batch_map_mean(self.train_function, train_iter, n_batches=n_batches)
            train_loss, unsup_loss, conf_rate, mask_rate = train_res[0], train_res[1], train_res[-2], train_res[-1]

            if conf_rate > best_mask_rate:
                best_mask_rate, improve = conf_rate, '*** '
                best_state = {k: v.cpu().numpy() for k, v in self.teacher_net.state_dict().items()}
            else:
                improve = ''

            src_stu_err, src_tea_err = source_test.batch_map_mean(self.evaluate_source_data, batch_size=batch_sz * 2)
            tgt_stu_err, tgt_tea_err = target_test.batch_map_mean(self.evaluate_target_data, batch_size=batch_sz * 2)

            print(f'{improve}Epoch {epoch} took {time.time() - t_start:.2f}s: TRAIN clf loss={train_loss:.6f}, '
                f'unsup (tgt) loss={unsup_loss:.6f}, conf mask={conf_rate:.3%}, unsup mask={mask_rate:.3%}; '
                f'SRC TEST ERR={src_stu_err:.3%}, TGT TEST student err={tgt_stu_err:.3%}, TGT TEST teacher err={tgt_tea_err:.3%}')

        self.teacher_net.load_state_dict({k: torch.from_numpy(v) for k, v in best_state.items()})
        self.teacher_net.eval()
        self.teacher_net.to(device)
        return self.teacher_net

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
        confidence_threshold = 0.96837722

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

        return float(clf_loss) * n_samples, float(unsup_loss) * n_samples

    def predict_source_data(self, X_source):
        """
        Predicts the output probabilities for source data.
        """
        X_variable = torch.tensor(X_source, dtype=torch.float, device=self.device)
        self.student_net.eval()
        self.teacher_net.eval()
        return (F.softmax(self.student_net(X_variable), dim=1).detach().cpu().numpy(),
                F.softmax(self.teacher_net(X_variable), dim=1).detach().cpu().numpy())

    def predict_target_data(self, X_target):
        """
        Predicts the output probabilities for target data.
        """
        X_variable = torch.tensor(X_target, dtype=torch.float, device=self.device)
        self.student_net.eval()
        self.teacher_net.eval()
        return (F.softmax(self.student_net(X_variable), dim=1).detach().cpu().numpy(),
                F.softmax(self.teacher_net(X_variable), dim=1).detach().cpu().numpy())

    def evaluate_source_data(self, X_source, y_source):
        """
        Evaluates the performance of the model on source data.
        """
        y_pred_prob_student, y_pred_prob_teacher = self.predict_source_data(X_source)
        y_pred_student = np.argmax(y_pred_prob_student, axis=1)
        y_pred_teacher = np.argmax(y_pred_prob_teacher, axis=1)
        return (float((y_pred_student != y_source).sum()), float((y_pred_teacher != y_source).sum()))

    def evaluate_target_data(self, X_target, y_target):
        """
        Evaluates the performance of the model on target data.
        """
        y_pred_prob_student, y_pred_prob_teacher = self.predict_target_data(X_target)
        y_pred_student = np.argmax(y_pred_prob_student, axis=1)
        y_pred_teacher = np.argmax(y_pred_prob_teacher, axis=1)
        return (float((y_pred_student != y_target).sum()), float((y_pred_teacher != y_target).sum()))

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
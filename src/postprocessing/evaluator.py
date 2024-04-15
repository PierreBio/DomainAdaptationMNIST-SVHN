import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the SVHN test set.
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Precision (macro avg): {precision*100:.2f}%')
    print(f'Recall (macro avg): {recall*100:.2f}%')
    print(f'F1 Score (macro avg): {f1*100:.2f}%')

    cm = confusion_matrix(all_labels, all_predictions)
    print('Confusion Matrix:\n', cm)

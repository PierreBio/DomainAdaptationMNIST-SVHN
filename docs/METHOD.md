< Back to [README](../README.md)

# Method

We used the Mean teacher algorithm which is a semi-supervised technique dedicated here to make domain adaptation from MNIST to SVHN. The key idea behind this algorithm is to train model on labeled data from MNIST and unlabeled data from SVHN. For that, it creates a student model and a teacher model which are CNN classifiers. Then it leverages the consistency loss between the predictions of a student model and a teacher model to improve the quality of the learned representations.

## Detailed Steps

### Preprocessing
- MNIST Dataset Preprocessing and Augmentation
  1. Resizing: We increase the size of MNIST images to 32x32 pixels to match the SVHN image dimensions. This is crucial for ensuring that the model can be trained interchangeably on both datasets without needing to adjust the input size.
  2. Grayscale to RGB Conversion: We convert the single-channel grayscale MNIST images to three channels to match the RGB SVHN images. This is done by duplicating the grayscale values across all three channels.
  3. Augmentation: The MNIST dataset undergoes several augmentations:
       - RandomHorizontalFlip and RandomRotation: These augmentations introduce spatial variability, making the model more robust to position and orientation changes.
       - ColorJitter: Even if MNIST images are grayscale, adding color jittering simulates slight variations in lighting and color that could make the model more adaptable to the color SVHN images.
       - Normalization: Normalizing the images with mean and standard deviation values of 0.5 across all channels standardizes the data, aiding in model training convergence.
  4. SVHN Dataset Preprocessing
       - Normalization: Similar to the MNIST dataset, SVHN images are normalized with mean and standard deviation values of 0.5. This ensures that pixel values are within a similar range for both datasets, which is important for model training.

### Models preparation 
- We create two convolutional classifiers: a student model and a teacher model. Initially, these two models have the same architecture and parameters, except Student having Adam optimizer and Teacher having Weighted Avarage Optimizer.
 
### Data Source and Batch Preparation
- Supervised and Target Datasets: We create ArrayDataSource objects for both supervised (source MNIST) and target (unlabeled SVHN) datasets.
- We apply here again data augmentation on our two datasets (train MNIST with labels and train SVHN without labels):
  1. Translation: We introduce random translations to the images based on a translation_range. This can help the model become invariant to the position of the digits in the images, which is useful given the different contexts in which digits appear in the MNIST and SVHN datasets.
  2. Intensity Scaling: We randomly adjust the intensity scale of the images within specified bounds (intensity_scale_lower, intensity_scale_upper). This method can make the model more robust to variations in lighting and contrast. It can help to adapt from MNIST's simple background to the complex backgrounds in SVHN.
  3. Intensity Offset: We add a random offset to the intensity values (intensity_offset_lower, intensity_offset_upper) which can also further help the model to handle different lighting conditions and backgrounds.
  4. Noise Addition: We introduce a Gaussian noise with a standard deviation of noise_std_dev. It can help the model to become more robust to small perturbations, mimicking real-world imperfections in images.
  5. Affine Transformation: We use cv2.warpAffine to apply the translations and centered transformations to augment the spatial characteristics of the images on every color channel and we apply the transformation individually; this ensures the spatial consistency is maintained across the channels.

### Training Loop
- Model Preparation: We set the student and teacher networks to training mode with .train(). This is important for layers like dropout and batch normalization that have distinct behaviors during training and evaluation.
- Forward Pass:
    1. Source Domain: We pass the source data through the student network to obtain logits, which are used with the ground truth labels to compute the classification loss (clf_loss).
    2. Target Domain: We pass independently the target data through both the student and teacher networks. The softmax probabilities from these logits are then used to compute the unsupervised loss, focusing on aligning the student's predictions with the teacher's on the target domain.
- Loss Calculation:
    1. Supervised Loss: We use self.classification_criterion (typically cross-entropy) to calculate the loss between the student's predictions on the source domain and the true labels.
    2. Unsupervised Loss: We calculate it by compute_augmentation_loss function, which measures the difference between the student's and teacher's predictions on the target domain, adjusted by a confidence threshold and class balance considerations.
- Backpropagation and Optimization:
    1. We perform backpropagation (loss_expression.backward()) on the combined supervised and weighted unsupervised loss.
    2. We update the student model using its Weighted Average optimizer.
    3. The teacher model's parameters are updated with the custom teacher_optimizer.update_parameters() method which uses the exponential moving average (EMA) update rule: $\theta' = \alpha \theta' + (1 - \alpha) \theta$

- Loss Reporting: We use supervised and unsupervised losses for monitoring only.

- **Epochs and Tracking:** We run the training for X epochs, and we try to track various metrics including train loss, unsupervised (target) loss, confidence rate, and mask rate. We try to identify the best performing epoch based on the confidence rate and save the state of the teacher network accordingly.

### Evaluation and Model Selection
- Best State Selection: The selection of the best model state is based on the confidence rate (how confident the model makes predictions on unlabeled data). It focus on models that are more certain of their predictions on the target dataset.
- Test on SVHN dataset: Our final evaluation function calculates the global student model accuracy by accumulating the total number of correctly predicted labels and the total number of examples to calculate the overall accuracy of the model on the test dataset.

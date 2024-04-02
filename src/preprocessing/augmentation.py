import numpy as np
import cv2

def create_identity_transforms(num_transforms):
    """
    Create a batch of N identity transformation matrices of shape 2x3.
    """
    identity_matrices = np.zeros((num_transforms, 2, 3), dtype=np.float32)
    identity_matrices[:, 0, 0] = identity_matrices[:, 1, 1] = 1.0
    return identity_matrices

def multiply_transformation_matrices(matrix_a, matrix_b):
    """
    Multiply corresponding 2x3 transformation matrices from two batches.
    """
    two_by_two_a = matrix_a[:, :, :2]
    two_by_two_b = matrix_b[:, :, :2]

    translation_a = matrix_a[:, :, 2:3]
    translation_b = matrix_b[:, :, 2:3]

    product_two_by_two = np.matmul(two_by_two_a, two_by_two_b)
    product_translation = translation_a + np.matmul(two_by_two_a, translation_b)
    return np.append(product_two_by_two, product_translation, axis=2)

def generate_rotation_matrices(angles):
    """
    Generate a batch of 2x3 rotation matrices based on the provided angles.
    """
    num_angles = angles.shape[0]
    rotation_matrices = np.zeros((num_angles, 2, 3), dtype=np.float32)
    rotation_matrices[:, 0, 0] = rotation_matrices[:, 1, 1] = np.cos(angles)
    rotation_matrices[:, 1, 0] = np.sin(angles)
    rotation_matrices[:, 0, 1] = -np.sin(angles)
    return rotation_matrices

def center_transformations(transformations, image_size):
    """
    Adjust the given transformation matrices to re-center them around the origin (0,0),
    assuming the original center is at the midpoint of an image with the specified size.
    """
    height, width = image_size

    shift_to_origin = np.zeros((1, 2, 3), dtype=np.float32)
    shift_to_origin[0, 0, 0] = shift_to_origin[0, 1, 1] = 1.0
    shift_to_origin[0, 0, 2] = -width * 0.5
    shift_to_origin[0, 1, 2] = -height * 0.5

    centered_transformations = multiply_transformation_matrices(transformations, shift_to_origin)

    centered_transformations[:, 0, 2] += width * 0.5
    centered_transformations[:, 1, 2] += height * 0.5

    return centered_transformations


class AugmentationSettings(object):
    def __init__(self, translation_range,
                 intensity_scale_lower=None, intensity_scale_upper=None,
                 intensity_offset_lower=None, intensity_offset_upper=None,
                 noise_std_dev=0.0):
        """
        Initialize advanced image augmentation settings.
        """
        self.translation_range = translation_range
        self.intensity_scale_lower = intensity_scale_lower
        self.intensity_scale_upper = intensity_scale_upper
        self.intensity_offset_lower = intensity_offset_lower
        self.intensity_offset_upper = intensity_offset_upper
        self.noise_std_dev = noise_std_dev

    def augment(self, images):
        """
        Apply the defined augmentations to a batch of images.
        """
        images_copy = images.copy()
        identity_transforms = create_identity_transforms(len(images_copy))

        if self.translation_range > 0.0:
            translations = np.random.uniform(low=-self.translation_range, high=self.translation_range, size=(len(images_copy), 2, 1))
            identity_transforms[:, :, 2:] += translations

        if self.intensity_scale_lower is not None:
            scale_factors = np.random.uniform(low=self.intensity_scale_lower, high=self.intensity_scale_upper,
                                              size=(len(images_copy), 1, 1, 1))
            images_copy = (images_copy * scale_factors).astype(np.float32)

        if self.intensity_offset_lower is not None:
            offsets = np.random.uniform(low=self.intensity_offset_lower, high=self.intensity_offset_upper,
                                        size=(len(images_copy), 1, 1, 1))
            images_copy += offsets

        centered_transforms = center_transformations(identity_transforms, images_copy.shape[2:])
        for i in range(len(images_copy)):
            for c in range(images_copy.shape[1]):
                images_copy[i, c, :, :] = cv2.warpAffine(images_copy[i, c, :, :], centered_transforms[i, :, :], (images_copy.shape[3], images_copy.shape[2]))

        if self.noise_std_dev > 0.0:
            images_copy += np.random.normal(scale=self.noise_std_dev, size=images_copy.shape).astype(np.float32)

        return images_copy

    def augment_pair(self, images):
        """
        Apply the augmentation to a pair of images separately, ensuring the same transformation is applied to both.
        """
        return self.augment(images), self.augment(images)

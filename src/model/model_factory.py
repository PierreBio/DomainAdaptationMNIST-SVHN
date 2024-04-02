from src.model.cnn import CNN
from src.model.gan import GAN
from src.model.mean_teacher import MeanTeacher

def model_factory(model_type, **kwargs):
    if model_type == 'cnn':
        return CNN(**kwargs)
    elif model_type == 'gan':
        return GAN(**kwargs)
    else:
        return MeanTeacher(**kwargs)
        raise ValueError("Unknown model type: {}".format(model_type))
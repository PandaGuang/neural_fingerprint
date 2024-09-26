import torch
import torchvision.models as models
from utils.config import config
import copy


def load_pretrained_model(model_name, pretrained=True):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    model.eval()
    return model


def get_independently_trained_models():
    models_conf = config.get('models')
    model_names = models_conf.get('architectures')
    count = models_conf.get('independent_model_count')
    pretrained = models_conf.get('pretrained', True)

    independent_models = []
    for name in model_names:
        for i in range(count):
            model = load_pretrained_model(model_name=name, pretrained=pretrained)
            # Initialize weights differently
            for param in model.parameters():
                if param.requires_grad:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
            independent_models.append(model)
    return independent_models

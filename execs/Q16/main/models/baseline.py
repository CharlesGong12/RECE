from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import clip

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

resnet_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model_imagenet(num_classes, language_model, device,
                              feature_extraction_forward=False, feature_extract=True):
    model_ft, input_size = initialize_model_imagenet_(num_classes,
                                                      feature_extract=feature_extract,
                                                      use_pretrained=True,
                                                      feature_extraction_forward=feature_extraction_forward)
    return model_ft, input_size


def initialize_model_imagenet_(num_classes, feature_extract, use_pretrained=True,
                               feature_extraction_forward=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    if feature_extraction_forward:
        model_ft.fc = nn.Identity()
    else:
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224
    transform = resnet_transforms['test']
    model_ft.preprocess = lambda x: transform(x)
    model_ft.encode = lambda x: model_ft.forward(x)
    model_ft.loss = _model_loss_placeholder
    return model_ft, input_size


def _model_loss_placeholder():
    return 0


def get_transforms(input_size):
    return resnet_transforms['train'], resnet_transforms['test']

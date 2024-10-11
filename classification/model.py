import torch
import torchvision.models as models

def get_model(model_name, num_classes):
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

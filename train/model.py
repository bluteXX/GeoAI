import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def get_model(num_classes):
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, num_classes),
    )

    return model


if __name__ == "__main__":
    model = get_model(num_classes=10)
    print("Model loaded successfully.")
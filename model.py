import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model(num_classes):
    # 1. Pobieramy sieć
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    print(model)

    # 2. Zamrażamy warstwy (Fine-tuning początkowy)
    for param in model.parameters():
        param.requires_grad = False

    # 3. Podmieniamy ostatnią warstwę (Fully Connected)
    num_ftrs =model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model


if __name__ == "__main__":
    test_model = get_model(num_classes=5)
    print("Model gotowy. Ostatnia warstwa:", test_model.fc)
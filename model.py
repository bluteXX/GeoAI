import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_model(num_classes):
    # 1. Pobieramy ResNet-50 z wagami z ImageNet
    # To jest nasz 'Baseline' - inteligentny start.
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)

    # 2. Zamrażamy warstwy (Fine-tuning początkowy)
    # Nie chcemy na razie zmieniać wag ekstraktora cech
    for param in model.parameters():
        param.requires_grad = False

    # 3. Podmieniamy ostatnią warstwę (Fully Connected)
    # ResNet-50 ma na końcu 2048 cech wejściowych
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


if __name__ == "__main__":
    test_model = get_model(num_classes=5)
    print("Model gotowy. Ostatnia warstwa:", test_model.fc)
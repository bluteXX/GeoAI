import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

def get_model(num_classes):
    # 1. Pobieramy głębszą sieć (B3)
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)

    # 2. Zamrażamy warstwy bazowe
    for param in model.parameters():
        param.requires_grad = False

    # 3. Dodajemy mocny Dropout i podmieniamy klasyfikator
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),  # 40% neuronów wyłączane podczas treningu
        nn.Linear(num_ftrs, num_classes)
    )

    return model

if __name__ == "__main__":
    test_model = get_model(num_classes=10)
    print("Model załadowany pomyślnie!")
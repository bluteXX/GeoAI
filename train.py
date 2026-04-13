import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import setup_data


def train_model():
    # Parametry
    epochs = 15  # Zwiększamy, żeby augmentacja miała czas zadziałać
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Trening na urządzeniu: {device}")

    dataloaders, class_names = setup_data(batch_size=batch_size)
    model = get_model(len(class_names))
    model = model.to(device)

    # Modyfikacja nr 3: Mniejszy Learning Rate (Adam jest stabilniejszy)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'Epoch {epoch + 1}/{epochs} | {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Zapisujemy model po każdej epoce (jako backup)
        torch.save(model.state_dict(), "geoguesser_baseline.pth")


if __name__ == "__main__":
    train_model()
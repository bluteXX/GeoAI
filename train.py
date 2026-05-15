import torch
import torch.nn as nn
import torch.optim as optim
import copy  # Dodane do zapisu najlepszego modelu
from model import get_model
from dataset import setup_data
import torch.optim.lr_scheduler as lr_scheduler


def train_model():
    epochs = 20
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Trening na urządzeniu: {device}")

    dataloaders, class_names = setup_data(batch_size=batch_size)
    model = get_model(len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)

    print(f"Rozpoczęcie treningu na {epochs} epok...")

    # Zmienne do śledzenia najlepszego modelu
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        if epoch == 5:
            print("🔓 Odmrażanie reszty sieci! Przechodzimy w tryb głębokiego Fine-Tuningu.")
            for param in model.parameters():
                param.requires_grad = True

            # Zmiana z Adam na AdamW + start z nieco wyższego pułapu (1e-4)
            optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

            # Inicjalizacja Schedulera (restart co 5 epok)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0


            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

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
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset)

            print(f'Epoch {epoch + 1}/{epochs} | {phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train' and epoch >= 5:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"📉 Aktualny Learning Rate: {current_lr:.7f}")
                scheduler.step()

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"🌟 Znaleziono nowy najlepszy model! (Zapisuję)")
                torch.save(best_model_wts, "geoguesser_baseline.pth")

    print(f"Trening zakończony. Najwyższa dokładność walidacyjna: {best_acc:.4f}")


if __name__ == "__main__":
    train_model()
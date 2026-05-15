import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from model import get_model
from dataset import setup_data
import torch.optim.lr_scheduler as lr_scheduler


def train_model():
    epochs = 50
    batch_size = 16
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Trening na urządzeniu: {device}")

    dataloaders, class_names = setup_data(batch_size=batch_size)
    model = get_model(len(class_names)).to(device)

    # Dodane Label Smoothing (zapobiega zbytniej pewności siebie modelu)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    patience = 8
    epochs_no_improve = 0

    print(f"Rozpoczęcie treningu na {epochs} epok...")

    try:
        for epoch in range(epochs):
            if epoch == 5:
                print("\n🔓 Odmrażanie reszty sieci! Przechodzimy w tryb głębokiego Fine-Tuningu.")
                for param in model.parameters():
                    param.requires_grad = True

                # Lekko podkręcony LR (z 1e-5 na 5e-5), szybsza nauka, ale wciąż bezpieczna dla wag
                optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
                # Podniesione eta_min do 1e-6, żeby pod koniec cyklu nie zwalniał aż do zera
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

                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        print(f"🌟 Znaleziono nowy najlepszy model! (Zapisuję)")
                        torch.save(best_model_wts, "geoguesser_baseline.pth")
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= patience and epoch > 7:
                            print(f"\n🛑 Wczesne zatrzymanie! Brak poprawy od {patience} epok.")
                            print(f"Trening zakończony. Najwyższa dokładność walidacyjna: {best_acc:.4f}")
                            return
    except KeyboardInterrupt:
        print("\nPrzerwano ręcznie! Zapisywanie najlepszego modelu przed wyjściem...")
        torch.save(best_model_wts, "geoguesser_baseline.pth")
        print(f"✅ Model zapisany. Najlepsza celność: {best_acc:.4f}")
        return

    print(f"\n✅ Trening zakończony. Najwyższa dokładność walidacyjna: {best_acc:.4f}")


if __name__ == "__main__":
    train_model()
import torch
import torch.nn as nn
import torch.optim as optim
from model import get_model
from dataset import setup_data
import torch.optim.lr_scheduler as lr_scheduler  # DODANE: potrzebne do sterowania LR


def train_model():
    # Parametry
    torch.backends.cudnn.benchmark = True
    epochs = 20  # Zwiększamy na 20, by Fine-Tuning miał czas zadziałać
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Trening na urządzeniu: {device}")

    dataloaders, class_names = setup_data(batch_size=batch_size)
    model = get_model(len(class_names))
    model = model.to(device)

    # Label Smoothing od razu dodany dla stabilności
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # FAZA 1: Podajemy do optymalizatora TYLKO niezamrożone warstwy (nową głowę)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=0.001)

    print(f"Rozpoczęcie treningu na {epochs} epok...")

    best_acc = 0.0  # Do śledzenia najlepszego wyniku

    for epoch in range(epochs):

        # --- START FINE-TUNING ---
        if epoch == 5:
            print("\n🔓 Odmrażanie reszty sieci! Przechodzimy w tryb głębokiego Fine-Tuningu.")
            # 1. Odmrażamy wszystkie parametry
            for param in model.parameters():
                param.requires_grad = True

            # 2. Zmieniamy optymalizator na AdamW z dużo mniejszym Learning Rate,
            #    aby nie "rozwalić" pre-trenowanych wag z ImageNetu.
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)

            # 3. Odpalamy harmonogram (Scheduler), który będzie powoli i faliście zmniejszał LR
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        # --- KONIEC FINE-TUNING ---

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

            # Aktualizujemy Scheduler tylko podczas treningu po 5 epoce
            if phase == 'train' and epoch >= 5:
                scheduler.step()

            # Zapisujemy model TYLKO po walidacji i TYLKO jeśli pobił rekord
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), "geoguesser_baseline.pth")
                    print(f"🌟 Znaleziono nowy najlepszy model! Zapisano na dysku. (Acc: {best_acc:.4f})")

    print(f"\n✅ Trening zakończony. Najwyższa celność walidacyjna: {best_acc:.4f}")


if __name__ == "__main__":
    train_model()
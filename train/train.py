import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from dataset import setup_data
from model import get_model

EPOCHS = 50
BATCH_SIZE = 16
PATIENCE = 8
UNFREEZE_EPOCH = 5
MODEL_OUTPUT_PATH = "geoguesser_baseline.pth"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unfreeze_model(model, device):
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    print(f"Model unfrozen. Fine-tuning started on {device}.")
    return optimizer, scheduler


def run_epoch(model, phase, dataloader, optimizer, criterion, device):
    model.train() if phase == 'train' else model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
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

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.float() / len(dataloader.dataset)
    return epoch_loss, epoch_acc


def train_model():
    device = get_device()
    print(f"Training on: {device}")

    dataloaders, class_names = setup_data(batch_size=BATCH_SIZE)
    model = get_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
    )
    scheduler = None

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    try:
        for epoch in range(EPOCHS):
            if epoch == UNFREEZE_EPOCH:
                optimizer, scheduler = unfreeze_model(model, device)

            for phase in ('train', 'val'):
                loss, acc = run_epoch(model, phase, dataloaders[phase], optimizer, criterion, device)
                print(f"Epoch {epoch + 1}/{EPOCHS} | {phase.capitalize()} Loss: {loss:.4f} Acc: {acc:.4f}")

                if phase == 'train' and scheduler is not None:
                    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.7f}")
                    scheduler.step()

                if phase == 'val':
                    if acc > best_acc:
                        best_acc = acc
                        best_weights = copy.deepcopy(model.state_dict())
                        torch.save(best_weights, MODEL_OUTPUT_PATH)
                        print(f"New best model saved (val acc: {best_acc:.4f}).")
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= PATIENCE and epoch > 7:
                            print(f"Early stopping after {PATIENCE} epochs without improvement.")
                            print(f"Best validation accuracy: {best_acc:.4f}")
                            return

    except KeyboardInterrupt:
        print("Training interrupted. Saving best model...")
        torch.save(best_weights, MODEL_OUTPUT_PATH)
        print(f"Model saved. Best accuracy: {best_acc:.4f}")
        return

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    train_model()
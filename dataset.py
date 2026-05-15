# dataset.py
import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

TARGET_COUNTRIES = ['PL', 'CZ', 'GR']

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def setup_data(batch_size=16):
    print("⏳ Przygotowywanie danych...")
    path = kagglehub.dataset_download("sylshaw/streetview-by-country")
    data_dir = os.path.join(path, "streetview_images")

    # Bazowe zestawy
    full_dataset = datasets.ImageFolder(root=data_dir)
    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    new_class_to_idx = {country: i for i, country in enumerate(TARGET_COUNTRIES)}

    # Filtrowanie
    filtered_samples = []
    for s_path, old_label in full_dataset.samples:
        country_name = idx_to_class[old_label].upper()
        if country_name in TARGET_COUNTRIES:
            filtered_samples.append((s_path, new_class_to_idx[country_name]))

    # Podział indeksów (70% train, 15% val, 15% test)
    num_samples = len(filtered_samples)
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size

    indices = torch.arange(num_samples)
    train_idx, val_idx, test_idx = random_split(
        indices, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Tworzenie Subsetów z odpowiednimi transformacjami
    # Musimy stworzyć dwa obiekty ImageFolder, żeby mieć różne transformacje
    train_base = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    val_test_base = datasets.ImageFolder(root=data_dir, transform=data_transforms['val_test'])

    # Nadpisujemy próbki
    train_base.samples = filtered_samples
    val_test_base.samples = filtered_samples

    train_ds = Subset(train_base, train_idx)
    val_ds = Subset(val_test_base, val_idx)
    test_ds = Subset(val_test_base, test_idx)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    }

    return dataloaders, TARGET_COUNTRIES
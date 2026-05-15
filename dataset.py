import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# Konfiguracja - 10 państw
TARGET_COUNTRIES = ['PL', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'CZ', 'SE']

# Zaktualizowane transformacje
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(300, scale=(0.7, 1.0)), # Rozmiar dla B3
        transforms.RandomHorizontalFlip(p=0.2), # Zmniejszone ryzyko odwrócenia napisów
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Wymazywanie losowych fragmentów (2-10%)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val_test': transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300), # Rozmiar dla B3
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def setup_data(batch_size=16):
    print("⏳ Pobieranie/Ładowanie danych...")
    path = kagglehub.dataset_download("sylshaw/streetview-by-country")
    data_dir = os.path.join(path, "streetview_images")

    # Tworzymy bazy datasetów
    train_full = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    val_test_full = datasets.ImageFolder(root=data_dir, transform=data_transforms['val_test'])

    idx_to_class = {v: k for k, v in train_full.class_to_idx.items()}
    new_class_to_idx = {country: i for i, country in enumerate(TARGET_COUNTRIES)}

    # Filtrowanie
    filtered_samples = []
    for s_path, old_label in train_full.samples:
        country_name = idx_to_class[old_label].upper()
        if country_name in TARGET_COUNTRIES:
            filtered_samples.append((s_path, new_class_to_idx[country_name]))

    # Nadpisujemy próbki
    train_full.samples = filtered_samples
    val_test_full.samples = filtered_samples

    # Podział (80% Train, 20% Reszta)
    num_samples = len(filtered_samples)
    train_size = int(0.8 * num_samples)
    remainder_size = num_samples - train_size

    indices = torch.arange(num_samples)
    train_idx, remainder_idx = random_split(
        indices, [train_size, remainder_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Dzielimy 20% reszty na Val (10%) i Test (10%)
    val_size = remainder_size // 2
    test_size = remainder_size - val_size
    val_idx, test_idx = random_split(
        remainder_idx, [val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Tworzymy Subsets
    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_test_full, val_idx)
    test_ds = Subset(val_test_full, test_idx)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    }

    return dataloaders, TARGET_COUNTRIES
import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Konfiguracja
TARGET_COUNTRIES = ['DE', 'ES', 'BO']

# Modyfikacja nr 2: Augmentacja danych
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((512, 512)),
        # Losowe odbicie lustrzane (geografia często jest symetryczna)
        transforms.RandomHorizontalFlip(p=0.5),
        # Zmiana jasności, kontrastu i nasycenia (różne pory dnia/kamery)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Losowe lekkie obroty (symulacja nierównego terenu)
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


def setup_data(batch_size=16):
    print("⏳ Pobieranie/Ładowanie danych...")
    path = kagglehub.dataset_download("sylshaw/streetview-by-country")
    data_dir = os.path.join(path, "streetview_images")

    # Tworzymy dwa obiekty datasetu, żeby mieć różne transformacje
    train_full = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    val_full = datasets.ImageFolder(root=data_dir, transform=data_transforms['val'])

    idx_to_class = {v: k for k, v in train_full.class_to_idx.items()}
    new_class_to_idx = {country: i for i, country in enumerate(TARGET_COUNTRIES)}

    # Filtrowanie próbek i zmiana etykiet
    def get_filtered_samples(dataset_samples):
        filtered = []
        for s_path, old_label in dataset_samples:
            country_name = idx_to_class[old_label]
            if country_name.upper() in [c.upper() for c in TARGET_COUNTRIES]:
                new_label = new_class_to_idx[country_name.upper()]
                filtered.append((s_path, new_label))
        return filtered

    new_samples = get_filtered_samples(train_full.samples)

    # Nadpisujemy próbki w obu datasetach
    train_full.samples = new_samples
    train_full.imgs = new_samples
    val_full.samples = new_samples
    val_full.imgs = new_samples

    # Podział na te same indeksy dla obu
    train_size = int(0.8 * len(new_samples))
    val_size = len(new_samples) - train_size
    indices = torch.arange(len(new_samples))
    train_idx, val_idx = random_split(indices, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    # Tworzymy Subsets z odpowiednimi transformacjami
    train_ds = torch.utils.data.Subset(train_full, train_idx)
    val_ds = torch.utils.data.Subset(val_full, val_idx)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    }

    return dataloaders, TARGET_COUNTRIES
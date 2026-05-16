import os
import torch
import kagglehub
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Konfiguracja
TARGET_COUNTRIES = [
    'AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE',
]

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), # To musi być tutaj!
        # WANDALIZM: zamazujemy od 2% do 10% zdjęcia w locie
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value='random'),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Wspólne transformacje bez sztucznego psucia dla Val i Test
    'val_test': transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def setup_data(batch_size=16):
    print("⏳ Pobieranie/Ładowanie danych...")
    path = kagglehub.dataset_download("sylshaw/streetview-by-country")
    data_dir = os.path.join(path, "streetview_images")

    # Tworzymy obiekty datasetu z odpowiednimi transformacjami
    train_full = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    val_test_full = datasets.ImageFolder(root=data_dir, transform=data_transforms['val_test'])

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

    # Nadpisujemy próbki
    train_full.samples = new_samples
    train_full.imgs = new_samples
    val_test_full.samples = new_samples
    val_test_full.imgs = new_samples

    # PODZIAŁ NA 3 ZBIORY (80% Train, 10% Val, 10% Test)
    num_samples = len(new_samples)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    test_size = num_samples - train_size - val_size

    indices = torch.arange(num_samples)
    train_idx, val_idx, test_idx = random_split(
        indices,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Tworzymy Subsets z odpowiednimi transformacjami
    train_ds = torch.utils.data.Subset(train_full, train_idx)
    val_ds = torch.utils.data.Subset(val_test_full, val_idx)
    test_ds = torch.utils.data.Subset(val_test_full, test_idx)

    # Mamy 3 pełnoprawne loadery!
    dataloaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    }

    return dataloaders, TARGET_COUNTRIES
import os
import torch
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from dataset import setup_data

# =====================================================================
# KONFIGURACJA MODELI DO TESTOWANIA
# Upewnij się, że nazwy plików .pth odpowiadają Twoim zapisanym wagom!
# =====================================================================
MODELS_CONFIG = {
    'EfficientNet-B0': {
        'path': 'modele/geoguesser_b0.pth', # ZMIEŃ NA SWOJĄ NAZWĘ PLIKU
        'arch': 'b0'
    },
    'EfficientNet-B2': {
        'path': 'modele/geoguesser_b2.pth', # ZMIEŃ NA SWOJĄ NAZWĘ PLIKU
        'arch': 'b2'
    },
    'EfficientNet-B3': {
        'path': 'modele/geoguesser_b3.pth', # ZMIEŃ NA SWOJĄ NAZWĘ PLIKU
        'arch': 'b3'
    },
    'ResNet-50': {
        'path': 'modele/geoguesser_resnet.pth', # ZMIEŃ NA SWOJĄ NAZWĘ PLIKU
        'arch': 'resnet'
    }
}

REPORTS_DIR = "raport_koncowy"

def get_eval_model(arch, num_classes):
    """Funkcja budująca architekturę na podstawie nazwy, żeby załadować wagi."""
    if arch == 'b2':
        model = models.efficientnet_b2()
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, num_classes)
        )
    elif arch == 'b3':
        model = models.efficientnet_b3()
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, num_classes)
        )
    elif arch == 'b0':
        model = models.efficientnet_b0()
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, num_classes)
        )
    elif arch == 'resnet':
        # Zakładam standardowego ResNet-50
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Nieznana architektura: {arch}")

    return model

def evaluate_model(model_name, config, test_loader, class_names, device):
    print(f"\n[{model_name}] Rozpoczynam ocenę...")

    if not os.path.exists(config['path']):
        print(f" Brak pliku z wagami: {config['path']}. Pomijam ten model.")
        return None

    # Budowanie modelu i ładowanie wag
    model = get_eval_model(config['arch'], len(class_names))
    model.load_state_dict(torch.load(config['path'], map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i+1) % 10 == 0:
                print(f"  Przetworzono paczkę {i+1}/{len(test_loader)}")

    # Obliczanie metryk
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Generowanie Macierzy Pomyłek (Wykres)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Macierz Pomyłek - {model_name}\nDokładność: {acc*100:.1f}%", fontsize=14)
    plt.ylabel('Faktyczny Kraj', fontsize=12)
    plt.xlabel('Predykcja AI', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{model_name.lower().replace('-', '_')}.png"))
    plt.close()

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f" Generator Raportów na urządzeniu: {device}")

    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)

    # Ładowanie tylko zbioru testowego (10% danych, których modele NIGDY nie widziały)
    dataloaders, class_names = setup_data(batch_size=32)
    test_loader = dataloaders['test']

    print(f"\nZbiór testowy zawiera {len(test_loader.dataset)} zdjęć do weryfikacji.")

    results = {}

    for model_name, config in MODELS_CONFIG.items():
        metrics = evaluate_model(model_name, config, test_loader, class_names, device)
        if metrics:
            results[model_name] = metrics

    if not results:
        print("\n Nie udało się przetestować żadnego modelu. Sprawdź ścieżki do plików .pth!")
        return

    # Generowanie Wykresu Porównawczego
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=['#4C72B0', '#DD8452', '#55A868'][:len(names)])
    plt.title("Porównanie Skuteczności Modeli (Zbiór Testowy)", fontsize=16, fontweight='bold')
    plt.ylabel("Dokładność (Accuracy) [%]", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Dodawanie wartości nad słupkami
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "porownanie_modeli_wykres.png"))
    plt.close()

    # Generowanie Raportu Tekstowego (Markdown)
    report_path = os.path.join(REPORTS_DIR, "RAPORT_PREZENTACJA.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Raport z Ewaluacji Modeli GeoGuessr AI\n\n")
        f.write("Poniższe wyniki zostały wygenerowane na całkowicie odizolowanym zbiorze testowym, ")
        f.write("na którym modele nigdy wcześniej nie były uczone ani optymalizowane.\n\n")

        f.write("##  Tabela Wyników (Metryki)\n\n")
        f.write("| Model | Dokładność (Accuracy) | Precyzja (Precision) | Czułość (Recall) | F1-Score |\n")
        f.write("|-------|-----------------------|----------------------|------------------|----------|\n")

        for name, m in results.items():
            f.write(f"| **{name}** | {m['accuracy']*100:.2f}% | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n")

        f.write("\n\n##  Jak interpretować metryki na prezentacji?\n")
        f.write("- **Accuracy (Dokładność):** Jaki procent wszystkich zdjęć model zgadł poprawnie.\n")
        f.write("- **Precision (Precyzja):** Kiedy model mówi, że to Polska, to na ile procent faktycznie ma rację.\n")
        f.write("- **Recall (Czułość):** Ile z wszystkich faktycznych zdjęć Polski model potrafił odnaleźć.\n")
        f.write("- **F1-Score:** Średnia harmoniczna precyzji i czułości (najlepsza metryka, gdyby niektóre kraje miały mniej zdjęć).\n")

    print(f"\n✅ SUKCES! Raport i wykresy zostały zapisane w folderze: '{REPORTS_DIR}'")

if __name__ == '__main__':
    main()
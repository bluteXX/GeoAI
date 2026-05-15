import os
import random
from pathlib import Path

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from dataset import TARGET_COUNTRIES
from model import get_model


# ---------------------------------------------------------------------------
# Stałe
# ---------------------------------------------------------------------------

ANALYZED_COUNTRIES = ['PL', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'CZ', 'SE']
N_IMAGES = 200
BASE_DIR = Path(__file__).resolve().parent
MISTAKES_ROOT = BASE_DIR / "mistakes"
MODEL_PATH = "geoguesser_baseline.pth"
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_trained_model(model_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Wczytuje wagi modelu i przygotowuje go do inferencji i Grad-CAM."""
    print(f"🧠 Ładowanie modelu na: {device}...")
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Grad-CAM wymaga aktywnych gradientów
    for param in model.parameters():
        param.requires_grad = True

    return model


# ---------------------------------------------------------------------------
# Przetwarzanie obrazów
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str, size: tuple = IMAGE_SIZE):
    """Zwraca tensor wejściowy (1, C, H, W) oraz obraz RGB jako float32 [0, 1]."""
    img_pil = Image.open(image_path).convert("RGB").resize(size)
    rgb_img = np.array(img_pil).astype(np.float32) / 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    input_tensor = transform(img_pil).unsqueeze(0)

    return input_tensor, rgb_img


def run_inference(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device):
    """Przeprowadza inferencję i zwraca prawdopodobieństwa klas."""
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
    return probabilities


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def generate_heatmap(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer,
    target_class_idx: int,
    rgb_img: np.ndarray,
) -> np.ndarray:
    """Generuje wizualizację Grad-CAM dla wskazanej klasy."""
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


# ---------------------------------------------------------------------------
# Wizualizacja
# ---------------------------------------------------------------------------

def build_figure(
    rgb_img: np.ndarray,
    heatmap_img: np.ndarray,
    actual_country: str,
    pred_country: str,
    confidence: float,
    filename: str,
    results_dict: dict,
) -> plt.Figure:
    """Tworzy figurę z oryginalnym zdjęciem, heatmapą i wykresem klas."""
    is_correct = pred_country.upper() == actual_country.upper()
    result_color = "green" if is_correct else "red"
    status_msg = "TRAFIONY!" if is_correct else f"BŁĄD! (AI: {pred_country.upper()})"

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

    ax1.set_title(f"Faktyczny: {actual_country}\n{filename}", fontsize=10)
    ax1.imshow(rgb_img)
    ax1.axis("off")

    ax2.set_title(f"{status_msg}\nPewność: {confidence:.2f}%", color=result_color, fontweight="bold")
    ax2.imshow(heatmap_img)
    ax2.axis("off")

    countries = list(results_dict.keys())
    probs = [results_dict[c] * 100 for c in countries]
    ax3.barh(countries, probs, color="skyblue")
    ax3.set_xlim(0, 100)
    ax3.set_xlabel("Prawdopodobieństwo [%]")
    ax3.set_title("Analiza klas")

    plt.subplots_adjust(top=0.85, bottom=0.1, left=0.05, right=0.95, wspace=0.3)
    return fig


# ---------------------------------------------------------------------------
# Analiza jednego obrazu
# ---------------------------------------------------------------------------

def analyze_single_image(model: torch.nn.Module, data_dir: str, device: torch.device):
    """
    Losuje obraz, wykonuje predykcję i generuje wizualizację z Grad-CAM.
    Zwraca: (figura, słownik prawdopodobieństw, nazwa faktycznego kraju).
    """
    # Losowanie zdjęcia
    actual_country = random.choice(TARGET_COUNTRIES)
    country_folder = os.path.join(data_dir, actual_country)
    if not os.path.exists(country_folder):
        country_folder = country_folder.lower()

    filename = random.choice(os.listdir(country_folder))
    img_path = os.path.join(country_folder, filename)

    # Przetwarzanie i inferencja
    input_tensor, rgb_img = preprocess_image(img_path)
    probabilities = run_inference(model, input_tensor, device)
    pred_idx = int(np.argmax(probabilities))

    results_dict = {TARGET_COUNTRIES[i]: probabilities[i] for i in range(len(TARGET_COUNTRIES))}

    # Grad-CAM (ostatnia warstwa konwolucyjna EfficientNet)
    target_layer = model.features[-1]
    actual_country_idx = TARGET_COUNTRIES.index(actual_country.upper())
    heatmap = generate_heatmap(model, input_tensor.to(device), target_layer, actual_country_idx, rgb_img)

    fig = build_figure(
        rgb_img, heatmap,
        actual_country, TARGET_COUNTRIES[pred_idx],
        probabilities[pred_idx] * 100,
        filename, results_dict,
    )

    return fig, results_dict, actual_country


# ---------------------------------------------------------------------------
# Czyszczenie folderu z pomyłkami
# ---------------------------------------------------------------------------

def clear_mistakes_folder(folder: Path) -> None:
    """Usuwa wszystkie pliki z podfolderów folderu z pomyłkami."""
    for subfolder in folder.iterdir():
        for file in subfolder.iterdir():
            if file.is_file():
                file.unlink()


# ---------------------------------------------------------------------------
# Główna pętla
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    print("⏳ Pobieranie i weryfikacja danych z Kaggle (to potrwa tylko chwilę)...")
    kaggle_path = kagglehub.dataset_download("sylshaw/streetview-by-country")
    data_dir = os.path.join(kaggle_path, "streetview_images")

    model = load_trained_model(MODEL_PATH, len(TARGET_COUNTRIES), device)

    mistake_stats = {
        country: {other: 0 for other in ANALYZED_COUNTRIES if other != country}
        for country in ANALYZED_COUNTRIES
    }

    country_totals = {country: 0 for country in ANALYZED_COUNTRIES}

    # NOWE: Licznik wszystkich poprawnych odpowiedzi
    total_correct = 0

    clear_mistakes_folder(MISTAKES_ROOT)

    print(f"\n🚀 Rozpoczynam weryfikację {N_IMAGES} obrazów...")

    for i in range(N_IMAGES):
        fig, results, actual = analyze_single_image(model, data_dir, device)
        guessed = max(results, key=results.get)

        if actual in country_totals:
            country_totals[actual] += 1

        # Sprawdzamy, czy model zgadł poprawnie
        if actual == guessed:
            total_correct += 1
        else:
            is_mistake = actual in mistake_stats and guessed in mistake_stats.get(actual, {})
            if is_mistake:
                mistake_stats[actual][guessed] += 1
                mistake_dir = MISTAKES_ROOT / f"{actual}-{guessed}"
                mistake_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(mistake_dir / f"mistake_{i}.png")

        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"✅ Przeanalizowano {i + 1}/{N_IMAGES} zdjęć...")

    # NOWE: Wypisywanie całkowitego wyniku
    print("\n" + "=" * 60)
    overall_accuracy = (total_correct / N_IMAGES) * 100
    print(f"🎯 CAŁKOWITA SKUTECZNOŚĆ MODELU: {overall_accuracy:.2f}% ({total_correct}/{N_IMAGES})")
    print("=" * 60)

    print("\n📊 SZCZEGÓŁOWE STATYSTYKI POMYŁEK:")
    print("-" * 60)
    for country, mistakes in mistake_stats.items():
        total_tests = country_totals[country]
        if total_tests == 0:
            continue

        active_mistakes = {k: v for k, v in mistakes.items() if v > 0}
        total_mistakes = sum(active_mistakes.values())
        accuracy = ((total_tests - total_mistakes) / total_tests) * 100

        print(f"🌍 {country} (Testowano: {total_tests} razy)")

        if not active_mistakes:
            print("   └─ 🌟 Perfekcyjnie! Brak pomyłek.")
        else:
            mistakes_str = ", ".join([f"{k}: {v}" for k, v in active_mistakes.items()])
            print(f"   └─ ❌ Mylono z: {mistakes_str}")
    print("-" * 60)

if __name__ == "__main__":
    main()
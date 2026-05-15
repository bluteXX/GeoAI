import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from pathlib import Path

from model import get_model
from dataset import TARGET_COUNTRIES, setup_data

MODEL_PATH = "geoguesser_baseline.pth"
IMAGE_SIZE = (300, 300)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_model(num_classes, device):
    model = get_model(num_classes)
    if os.path.exists(MODEL_PATH):
        print(f"🧠 Wczytywanie modelu z {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    else:
        print(f"⚠️  Brak pliku modelu! Upewnij się, że {MODEL_PATH} istnieje.")
        exit()
    model.to(device)
    model.eval()
    return model

def preprocess(img_path):
    img_pil = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(img_pil).unsqueeze(0), np.array(img_pil)

def play_game():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        dataloaders, _ = setup_data(batch_size=1)
        if 'test' in dataloaders:
            test_ds = dataloaders['test'].dataset
        else:
            test_ds = dataloaders['val'].dataset
    except Exception as e:
        print(f"❌ Błąd podczas ładowania danych: {e}")
        return

    model = load_model(len(TARGET_COUNTRIES), device)

    score_user = 0
    score_ai = 0
    rounds = 5

    print("\n" + "=" * 40)
    print("🌍 WITAJ W GEOGUESSER AI 1v1! 🌍")
    print("=" * 40)
    print(f"Dostępne kraje: {', '.join(TARGET_COUNTRIES)}")
    print("Wpisz kod kraju, aby zgadnąć. Powodzenia!\n")

    for r in range(1, rounds + 1):
        print(f"--- RUNDA {r}/{rounds} ---")

        idx = random.randint(0, len(test_ds) - 1)
        img_path, actual_label_idx = test_ds.dataset.samples[test_ds.indices[idx]]
        actual_country = TARGET_COUNTRIES[actual_label_idx].upper()

        input_tensor, original_img = preprocess(img_path)
        with torch.no_grad():
            output = model(input_tensor.to(device))
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            ai_pred_idx = torch.argmax(output, dim=1).item()

        ai_guess = TARGET_COUNTRIES[ai_pred_idx].upper()
        ai_confidence = probabilities[ai_pred_idx].item() * 100

        plt.figure(figsize=(10, 7))
        plt.imshow(original_img)
        plt.title(f"Runda {r}: Gdzie wykonano to zdjęcie?")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)

        user_guess = ""
        while user_guess not in TARGET_COUNTRIES:
            user_guess = input("Twój typ (kod ISO, np. PL): ").strip().upper()
            if user_guess not in TARGET_COUNTRIES:
                print(f"Błąd! Wybierz spośród: {', '.join(TARGET_COUNTRIES)}")

        print(f"\n🤖 AI mówi: {ai_guess} (pewność: {ai_confidence:.2f}%)")
        print(f"✅ Prawidłowa odpowiedź: {actual_country}")

        user_correct = (user_guess == actual_country)
        ai_correct = (ai_guess == actual_country)

        if user_correct:
            print("✨ Punkt dla CIEBIE!")
            score_user += 1

        if ai_correct:
            print("🤖 Punkt dla AI!")
            score_ai += 1

        if not user_correct and not ai_correct:
            print("❌ Nikt nie zgadł.")

        print(f"AKTUALNY WYNIK: TY {score_user} : {score_ai} AI\n")
        plt.close()

    print("=" * 40)
    print("KONIEC GRY!")
    if score_user > score_ai:
        print(f"🏆 WYNIK {score_user}:{score_ai} - GRATULACJE! Pokonałeś maszynę!")
    elif score_ai > score_user:
        print(f"💻 WYNIK {score_user}:{score_ai} - AI wygrywa. Trenuj dalej!")
    else:
        print(f"🤝 WYNIK {score_user}:{score_ai} - REMIS!")
    print("=" * 40)

if __name__ == "__main__":
    play_game()
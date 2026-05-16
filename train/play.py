import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TARGET_COUNTRIES, setup_data
from evaluate import MODELS_CONFIG, build_model
from explain import build_figure, generate_heatmap, preprocess_image


BASE_DIR = Path(__file__).resolve().parent.parent

def play_game():
    print("AI GeoGuessr")
    print("-" * 20)

    model_names = list(MODELS_CONFIG.keys())
    print("Wybierz model:")
    for i, name in enumerate(model_names, 1):
        print(f"{i}. {name}")

    try:
        choice = int(input("Twój wybór (1-4): "))
        if choice < 1 or choice > len(model_names):
            raise ValueError
    except ValueError:
        print("Niepoprawny wybór.")
        return

    selected_model_name = model_names[choice - 1]
    config = MODELS_CONFIG[selected_model_name]

    # Poprawione łączenie ścieżek za pomocą operatora '/'
    model_path = BASE_DIR / config['path']

    if not model_path.exists():
        print(f"Nie znaleziono pliku modelu: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    model = build_model(config['arch'], len(TARGET_COUNTRIES))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = True

    try:
        n_rounds = int(input("Podaj liczbę rund: "))
    except ValueError:
        print("Wprowadzono niepoprawną wartość.")
        return

    dataloaders, _ = setup_data(batch_size=1)
    test_ds = dataloaders['test'].dataset

    user_score = 0
    ai_score = 0

    for round_num in range(1, n_rounds + 1):
        print(f"\nRunda {round_num}/{n_rounds}")
        print("-" * 20)

        idx = random.randint(0, len(test_ds) - 1)
        img_path, actual_idx = test_ds.dataset.samples[test_ds.indices[idx]]
        actual_country = TARGET_COUNTRIES[actual_idx]

        input_tensor, rgb_img = preprocess_image(img_path)

        plt.figure(figsize=(8, 8))
        plt.imshow(rgb_img)
        plt.title(f"Runda {round_num}")
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.1)

        print(f"Dostępne kraje: {', '.join(TARGET_COUNTRIES)}")
        user_guess = input("Twoja odpowiedź: ").strip()
        plt.close()

        with torch.no_grad():
            output = model(input_tensor.to(device))
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        ai_guess = TARGET_COUNTRIES[pred_idx]
        ai_confidence = probs[pred_idx] * 100
        class_probs = {TARGET_COUNTRIES[k]: probs[k] for k in range(len(TARGET_COUNTRIES))}

        print("\nWyniki rundy:")
        print(f"Prawdziwy kraj: {actual_country}")
        print(f"Twoja odpowiedź: {user_guess}")
        print(f"Odpowiedź AI: {ai_guess} ({ai_confidence:.1f}%)")

        user_correct = user_guess.lower() == actual_country.lower()
        ai_correct = ai_guess.lower() == actual_country.lower()

        if user_correct and ai_correct:
            print("Remis.")
            user_score += 1
            ai_score += 1
        elif user_correct:
            print("Punkt dla Ciebie.")
            user_score += 1
        elif ai_correct:
            print("Punkt dla AI.")
            ai_score += 1
        else:
            print("Brak punktów.")

        if config['arch'] == 'resnet':
            target_layer = model.layer4[-1]
        else:
            target_layer = model.features[-1]

        heatmap = generate_heatmap(model, input_tensor.to(device), target_layer, pred_idx, rgb_img)
        fig = build_figure(rgb_img, heatmap, actual_country, ai_guess, ai_confidence, Path(img_path).name, class_probs)
        plt.show(block=True)

    print("\nPodsumowanie gry:")
    print("-" * 20)
    print(f"Twój wynik: {user_score}")
    print(f"Wynik AI: {ai_score}")


if __name__ == "__main__":
    play_game()
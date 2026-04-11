import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
import kagglehub
from PIL import Image


from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from model import get_model


def run_explanation():
    # 1. Konfiguracja urządzenia
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_COUNTRIES = ['DE', 'ES', 'BO']

    print(f"🧠 Ładowanie modelu na: {device}...")
    model = get_model(len(TARGET_COUNTRIES))
    # Wczytujemy Twoje wyuczone wagi
    model.load_state_dict(torch.load("geoguesser_baseline.pth", weights_only=True))
    model.to(device)
    model.eval()

    # Odmrażamy parametry, aby Grad-CAM mógł policzyć wpływ pikseli na wynik
    for param in model.parameters():
        param.requires_grad = True

    # Ostatnia warstwa splotowa dla ResNet-50
    target_layers = [model.layer4[-1]]

    # 2. Losowanie zdjęcia z dysku
    print("📂 Przeszukiwanie bazy zdjęć...")
    path = kagglehub.dataset_download("sylshaw/streetview-by-country")
    data_dir = os.path.join(path, "streetview_images")

    actual_country = random.choice(TARGET_COUNTRIES)
    country_path = os.path.join(data_dir, actual_country)

    # Obsługa małych/wielkich liter w nazwach folderów
    if not os.path.exists(country_path):
        actual_country = actual_country.lower()
        country_path = os.path.join(data_dir, actual_country)

    random_filename = random.choice(os.listdir(country_path))
    image_path = os.path.join(country_path, random_filename)

    print(f"🖼️ Analizuję: {actual_country.upper()} | {random_filename}")

    # 3. Przygotowanie zdjęcia (identycznie jak przy treningu)
    img_pil = Image.open(image_path).convert('RGB').resize((512, 512))
    rgb_img = np.array(img_pil).astype(np.float32) / 255

    input_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(img_pil).unsqueeze(0).to(device)

    # 4. Przewidywanie (Co AI faktycznie obstawia?)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probabilities, 0)

    predicted_country = TARGET_COUNTRIES[pred_idx.item()]
    confidence_score = conf.item() * 100

    # 5. Generowanie Heatmapy (Grad-CAM)
    target_idx = TARGET_COUNTRIES.index(actual_country.upper())
    targets = [ClassifierOutputTarget(target_idx)]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 6. Wyświetlanie wyników (Poprawione marginesy!)
    fig = plt.figure(figsize=(14, 8))

    is_correct = predicted_country.upper() == actual_country.upper()
    result_color = 'green' if is_correct else 'red'
    status_msg = "TRAFIONY!" if is_correct else f"BŁĄD! (AI myślało, że to {predicted_country.upper()})"

    # Panel 1: Oryginał
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title(f"Faktyczny kraj: {actual_country.upper()}\nPlik: {random_filename}", fontsize=10, pad=15)
    ax1.imshow(rgb_img)
    ax1.axis('off')

    # Panel 2: Heatmapa
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title(f"{status_msg}\nAI Pewność: {confidence_score:.2f}%", color=result_color, fontsize=15,
                  fontweight='bold', pad=15)
    ax2.imshow(visualization)
    ax2.axis('off')

    # Ręczne ustawienie marginesów, żeby nic nie ucięło (top=0.85 daje 15% miejsca na górze)
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.2)

    print("🚀 Wyświetlam okno graficzne...")
    plt.show()


if __name__ == "__main__":
    run_explanation()
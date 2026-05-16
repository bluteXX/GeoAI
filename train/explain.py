import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

from dataset import TARGET_COUNTRIES, setup_data

BASE_DIR = Path(__file__).resolve().parent.parent
MISTAKES_ROOT = BASE_DIR / "mistakes"
MODEL_PATH = BASE_DIR / "modele/geoguesser_b2.pth"

N_IMAGES = 1000
IMAGE_SIZE = (300, 300)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(model_path, num_classes, device):
    model = models.efficientnet_b2()
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    rgb_array = np.array(img).astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0), rgb_array


def generate_heatmap(model, input_tensor, target_layer, target_class_idx, rgb_img):
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class_idx)])[0]
    return show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)


def build_figure(rgb_img, heatmap_img, actual, predicted, confidence, filename, class_probs):
    color = "green" if predicted.upper() == actual.upper() else "red"
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 7))

    ax1.set_title(f"Actual: {actual}\n{filename}", fontsize=8)
    ax1.imshow(rgb_img)
    ax1.axis("off")

    ax2.set_title(f"Predicted: {predicted} ({confidence:.1f}%)", color=color, fontweight="bold")
    ax2.imshow(heatmap_img)
    ax2.axis("off")

    ax3.barh(list(class_probs.keys()), [v * 100 for v in class_probs.values()], color="skyblue")
    ax3.set_xlim(0, 100)
    ax3.set_title("Class probabilities [%]")

    plt.tight_layout()
    return fig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders, _ = setup_data(batch_size=1)
    test_ds = dataloaders['test'].dataset
    model = load_model(MODEL_PATH, len(TARGET_COUNTRIES), device)

    if MISTAKES_ROOT.exists():
        shutil.rmtree(MISTAKES_ROOT)
    MISTAKES_ROOT.mkdir(parents=True)

    correct_count = 0
    mistake_matrix = {c: {other: 0 for other in TARGET_COUNTRIES if other != c} for c in TARGET_COUNTRIES}

    n_test = min(N_IMAGES, len(test_ds))
    indices = random.sample(range(len(test_ds)), n_test)
    print(f"Analyzing {n_test} images from test set...")

    for i, idx in enumerate(indices):
        img_path, actual_idx = test_ds.dataset.samples[test_ds.indices[idx]]
        actual_country = TARGET_COUNTRIES[actual_idx]

        input_tensor, rgb_img = preprocess_image(img_path)

        with torch.no_grad():
            output = model(input_tensor.to(device))
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_country = TARGET_COUNTRIES[pred_idx]
        class_probs = {TARGET_COUNTRIES[k]: probs[k] for k in range(len(TARGET_COUNTRIES))}

        if pred_country == actual_country:
            correct_count += 1
            target_layer = model.features[-1]
            heatmap = generate_heatmap(model, input_tensor.to(device), target_layer, pred_idx, rgb_img)
            fig = build_figure(rgb_img, heatmap, actual_country, pred_country,
                               probs[pred_idx] * 100, Path(img_path).name, class_probs)

            err_dir = MISTAKES_ROOT / f"{actual_country}"
            err_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(err_dir / f"correct_{i}.png")
            plt.close(fig)
        else:
            mistake_matrix[actual_country][pred_country] += 1
            target_layer = model.features[-1]
            heatmap = generate_heatmap(model, input_tensor.to(device), target_layer, pred_idx, rgb_img)
            fig = build_figure(rgb_img, heatmap, actual_country, pred_country,
                               probs[pred_idx] * 100, Path(img_path).name, class_probs)

            err_dir = MISTAKES_ROOT / f"{actual_country}_as_{pred_country}"
            err_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(err_dir / f"error_{i}.png")
            plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_test}")

    accuracy = (correct_count / n_test) * 100
    print(f"\nTest accuracy: {accuracy:.2f}% ({correct_count}/{n_test})")

    print("\nMistake analysis:")
    for actual, mistakes in mistake_matrix.items():
        total = sum(mistakes.values())
        if total > 0:
            details = {k: v for k, v in mistakes.items() if v > 0}
            print(f"  {actual} confused with: {details}")


if __name__ == "__main__":
    main()
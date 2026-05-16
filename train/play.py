import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from dataset import TARGET_COUNTRIES, setup_data
from model import get_model

MODEL_PATH = "geoguesser_baseline.pth"
IMAGE_SIZE = (300, 300)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ROUNDS = 5


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(num_classes, device):
    model = get_model(num_classes)
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        sys.exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = True
    return model


def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMAGE_SIZE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(img).unsqueeze(0), np.array(img)


def get_top3_text(probs_np):
    top3 = np.argsort(probs_np)[-3:][::-1]
    return " | ".join([f"{TARGET_COUNTRIES[i]}: {probs_np[i] * 100:.1f}%" for i in top3])


def generate_gradcam(model, input_tensor, pred_idx, original_img, device):
    model_cpu = model.to("cpu")
    input_cpu = input_tensor.to("cpu")
    target_layer = model_cpu.features[-1] if hasattr(model_cpu, 'features') else model_cpu.layer4[-1]

    cam = GradCAM(model=model_cpu, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_cpu, targets=[ClassifierOutputTarget(pred_idx)])[0]
    rgb_float = original_img.astype(np.float32) / 255.0
    heatmap = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    model.to(device)
    return heatmap


def prompt_user_guess():
    while True:
        guess = input(f"Your guess ({', '.join(TARGET_COUNTRIES)}): ").strip().upper()
        if guess in TARGET_COUNTRIES:
            return guess
        print(f"Invalid input. Choose from: {', '.join(TARGET_COUNTRIES)}")


def show_round_result(round_num, original_img, heatmap, actual, user_guess, ai_guess,
                      user_correct, ai_correct, top3_text):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Round {round_num} — Answer: {actual}", fontsize=16, fontweight="bold")

    ax1.imshow(original_img)
    ax1.set_title(f"Your guess: {user_guess} | {'Correct' if user_correct else 'Wrong'}")
    ax1.axis('off')

    ax2.imshow(heatmap)
    ax2.set_title(f"AI guess: {ai_guess} | {'Correct' if ai_correct else 'Wrong'}")
    ax2.axis('off')

    plt.figtext(0.5, 0.05, f"AI Top 3: {top3_text}", ha="center", fontsize=12,
                bbox={"facecolor": "#3498db", "alpha": 0.2, "pad": 8, "boxstyle": "round,pad=0.5"})

    plt.show(block=False)
    plt.pause(0.1)


def play_game():
    device = get_device()

    try:
        dataloaders, _ = setup_data(batch_size=1)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    test_ds = dataloaders.get('test', dataloaders.get('val')).dataset
    model = load_model(len(TARGET_COUNTRIES), device)

    score_user, score_ai = 0, 0

    print(f"\n{'=' * 40}")
    print("GEOGUESSER AI — 1v1")
    print(f"{'=' * 40}")
    print(f"Countries: {', '.join(TARGET_COUNTRIES)}\n")

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Round {round_num}/{ROUNDS} ---")

        idx = random.randint(0, len(test_ds) - 1)
        img_path, actual_label_idx = test_ds.dataset.samples[test_ds.indices[idx]]
        actual_country = TARGET_COUNTRIES[actual_label_idx].upper()

        input_tensor, original_img = preprocess(img_path)
        output = model(input_tensor.to(device))
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        ai_pred_idx = torch.argmax(output, dim=1).item()
        ai_guess = TARGET_COUNTRIES[ai_pred_idx].upper()
        ai_confidence = probs[ai_pred_idx].item() * 100
        probs_np = probs.cpu().detach().numpy()
        top3_text = get_top3_text(probs_np)

        plt.figure(figsize=(8, 6))
        plt.imshow(original_img)
        plt.title(f"Round {round_num}/{ROUNDS}: Where was this photo taken?")
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.1)

        user_guess = prompt_user_guess()
        plt.close()

        print(f"AI: {ai_guess} (confidence: {ai_confidence:.2f}%)")
        print(f"Answer: {actual_country}")

        user_correct = user_guess == actual_country
        ai_correct = ai_guess == actual_country

        if user_correct:
            print("Point for you!")
            score_user += 1
        if ai_correct:
            print("Point for AI!")
            score_ai += 1
        if not user_correct and not ai_correct:
            print("Nobody guessed correctly.")

        print(f"Score: YOU {score_user} : {score_ai} AI\n")

        print("Generating Grad-CAM explanation...")
        heatmap = generate_gradcam(model, input_tensor, ai_pred_idx, original_img, device)
        show_round_result(round_num, original_img, heatmap, actual_country,
                          user_guess, ai_guess, user_correct, ai_correct, top3_text)

        input("Press ENTER to continue...")
        plt.close()

    print(f"\n{'=' * 40}")
    print("GAME OVER")
    if score_user > score_ai:
        print(f"Result {score_user}:{score_ai} — You win!")
    elif score_ai > score_user:
        print(f"Result {score_user}:{score_ai} — AI wins!")
    else:
        print(f"Result {score_user}:{score_ai} — Draw!")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    play_game()
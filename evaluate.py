import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torchvision import models

from dataset import setup_data

MODELS_CONFIG = {
    'EfficientNet-B0': {'path': 'modele/geoguesser_b0.pth', 'arch': 'b0'},
    'EfficientNet-B2': {'path': 'modele/geoguesser_b2.pth', 'arch': 'b2'},
    'EfficientNet-B3': {'path': 'modele/geoguesser_b3.pth', 'arch': 'b3'},
    'ResNet-50':        {'path': 'modele/geoguesser_resnet.pth', 'arch': 'resnet'},
}

REPORTS_DIR = "raport_koncowy"


def build_model(arch, num_classes):
    if arch in ('b0', 'b2', 'b3'):
        constructor = {'b0': models.efficientnet_b0, 'b2': models.efficientnet_b2, 'b3': models.efficientnet_b3}[arch]
        model = constructor()
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, num_classes),
        )
    elif arch == 'resnet':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


def evaluate_model(model_name, config, test_loader, class_names, device):
    print(f"Evaluating {model_name}...")

    if not os.path.exists(config['path']):
        print(f"Weights file not found: {config['path']}. Skipping.")
        return None

    model = build_model(config['arch'], len(class_names))
    model.load_state_dict(torch.load(config['path'], map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Batch {i + 1}/{len(test_loader)}")

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    _save_confusion_matrix(cm, class_names, model_name, acc)

    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def _save_confusion_matrix(cm, class_names, model_name, acc):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}\nAccuracy: {acc * 100:.1f}%", fontsize=14)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    filename = f"confusion_matrix_{model_name.lower().replace('-', '_')}.png"
    plt.savefig(os.path.join(REPORTS_DIR, filename))
    plt.close()


def _save_comparison_chart(results):
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] * 100 for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(names)])
    plt.title("Model Accuracy Comparison (Test Set)", fontsize=16, fontweight='bold')
    plt.ylabel("Accuracy [%]", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%",
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"))
    plt.close()


def _save_markdown_report(results):
    report_path = os.path.join(REPORTS_DIR, "REPORT.md")
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# GeoGuessr AI — Model Evaluation Report\n\n")
        f.write("Results generated on a fully isolated test set not used during training.\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        for name, m in results.items():
            f.write(
                f"| **{name}** | {m['accuracy'] * 100:.2f}% "
                f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |\n"
            )
        f.write("\n## Metric Definitions\n\n")
        f.write("- **Accuracy**: Percentage of correctly classified images.\n")
        f.write("- **Precision**: Of all images predicted as class X, how many were actually X.\n")
        f.write("- **Recall**: Of all images that are class X, how many were correctly identified.\n")
        f.write("- **F1-Score**: Harmonic mean of precision and recall.\n")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Device: {device}")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    dataloaders, class_names = setup_data(batch_size=32)
    test_loader = dataloaders['test']
    print(f"Test set size: {len(test_loader.dataset)} images.")

    results = {}
    for model_name, config in MODELS_CONFIG.items():
        metrics = evaluate_model(model_name, config, test_loader, class_names, device)
        if metrics:
            results[model_name] = metrics

    if not results:
        print("No models evaluated. Check .pth file paths.")
        return

    _save_comparison_chart(results)
    _save_markdown_report(results)
    print(f"Report saved to '{REPORTS_DIR}'.")


if __name__ == '__main__':
    main()
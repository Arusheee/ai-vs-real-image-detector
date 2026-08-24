import os
import torch
import timm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix

EXTERNAL_TEST_DIR = r"external-test-images"
MODEL_PATH = "best_modelNew.pt"
MODEL_ARCHITECTURE = "resnet18"
IMG_SIZE = 224
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_ARCHITECTURE, pretrained=False, num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {device}")
    return model, device


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_ai_probability(model, device, transform, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    return probs[1].item()


def collect_images(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]


def main():
    model, device = load_model()
    transform = get_transform()

    real_dir = os.path.join(EXTERNAL_TEST_DIR, "human")
    ai_dir = os.path.join(EXTERNAL_TEST_DIR, "ai")
    real_images = collect_images(real_dir)
    ai_images = collect_images(ai_dir)

    print(f"Found {len(real_images)} real images, {len(ai_images)} AI images\n")

    true_labels = []
    ai_probs = []

    for path in real_images:
        true_labels.append(0)
        ai_probs.append(get_ai_probability(model, device, transform, path))

    for path in ai_images:
        true_labels.append(1)
        ai_probs.append(get_ai_probability(model, device, transform, path))

    print("Threshold sweep (predict 'AI' if ai_probability > threshold):\n")
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'Real correct':>13} | {'AI correct':>11}")
    print("-" * 55)

    best_threshold = 0.5
    best_acc = 0.0

    for t in [round(x * 0.02 + 0.30, 2) for x in range(21)]:
        preds = [1 if p > t else 0 for p in ai_probs]
        acc = accuracy_score(true_labels, preds)
        cm = confusion_matrix(true_labels, preds, labels=[0, 1])
        real_correct = cm[0][0] / max(1, cm[0].sum())
        ai_correct = cm[1][1] / max(1, cm[1].sum())
        marker = ""
        if acc > best_acc:
            best_acc = acc
            best_threshold = t
            marker = "  <- best so far"
        print(f"{t:>10.2f} | {acc*100:>8.2f}% | {real_correct*100:>12.1f}% | {ai_correct*100:>10.1f}%{marker}")

    print(f"\nBest threshold: {best_threshold} (accuracy: {best_acc*100:.2f}%)")

    final_preds = [1 if p > best_threshold else 0 for p in ai_probs]
    final_cm = confusion_matrix(true_labels, final_preds, labels=[0, 1])
    print("\nConfusion matrix at best threshold:")
    print(final_cm)


if __name__ == "__main__":
    main()
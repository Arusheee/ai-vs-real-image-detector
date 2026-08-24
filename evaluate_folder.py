import os
import torch
import timm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

EXTERNAL_TEST_DIR = r"external-test-images"
MODEL_PATH = "best_modelNew.pt"
MODEL_ARCHITECTURE = "resnet18"
IMG_SIZE = 224
DECISION_THRESHOLD = 0.68
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


def predict_single(model, device, transform, image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    real_prob = probs[0].item()
    ai_prob = probs[1].item()
    pred_label = 1 if ai_prob > DECISION_THRESHOLD else 0
    confidence = max(real_prob, ai_prob)
    return pred_label, confidence


def collect_images(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext in VALID_EXTENSIONS:
            files.append(os.path.join(folder, f))
    return files


def main():
    model, device = load_model()
    transform = get_transform()

    real_dir = os.path.join(EXTERNAL_TEST_DIR, "human")
    ai_dir = os.path.join(EXTERNAL_TEST_DIR, "ai")

    real_images = collect_images(real_dir)
    ai_images = collect_images(ai_dir)

    if not real_images and not ai_images:
        print(f"No images found. Checked:\n  {real_dir}\n  {ai_dir}")
        return

    print(f"Found {len(real_images)} real images, {len(ai_images)} AI images\n")

    all_true = []
    all_pred = []
    all_conf = []
    misclassified = []

    print("Evaluating real images...")
    for path in real_images:
        pred, conf = predict_single(model, device, transform, path)
        all_true.append(0)
        all_pred.append(pred)
        all_conf.append(conf)
        status = "OK" if pred == 0 else "WRONG"
        print(f"  [{status}] {os.path.basename(path)} -> predicted {'AI' if pred==1 else 'Real'} ({conf*100:.1f}%)")
        if pred != 0:
            misclassified.append((path, "Real", "AI", conf))

    print("\nEvaluating AI images...")
    for path in ai_images:
        pred, conf = predict_single(model, device, transform, path)
        all_true.append(1)
        all_pred.append(pred)
        all_conf.append(conf)
        status = "OK" if pred == 1 else "WRONG"
        print(f"  [{status}] {os.path.basename(path)} -> predicted {'AI' if pred==1 else 'Real'} ({conf*100:.1f}%)")
        if pred != 1:
            misclassified.append((path, "AI", "Real", conf))

    acc = accuracy_score(all_true, all_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, average="binary", zero_division=0)
    cm = confusion_matrix(all_true, all_pred)
    avg_conf = sum(all_conf) / len(all_conf)

    print("\n" + "=" * 50)
    print("RESULTS — External Test Folder")
    print("=" * 50)
    print(f"Total images tested: {len(all_true)}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"Avg confidence: {avg_conf*100:.2f}%")
    print("\nConfusion matrix (rows=true, cols=predicted, 0=Real 1=AI):")
    print(cm)

    if misclassified:
        print(f"\n{len(misclassified)} misclassified image(s):")
        for path, true_label, pred_label, conf in misclassified:
            print(f"  - {os.path.basename(path)} | True: {true_label}, Predicted: {pred_label} ({conf*100:.1f}%)")
    else:
        print("\nNo misclassified images.")


if __name__ == "__main__":
    main()
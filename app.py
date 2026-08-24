import os
import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_modelNew.pt")
MODEL_ARCHITECTURE = "resnet18"
IMG_SIZE = 224
DECISION_THRESHOLD = 0.68

st.set_page_config(page_title="AI vs Human Image Detector", page_icon="🖼️", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(MODEL_ARCHITECTURE, pretrained=False, num_classes=2)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(model, device, image: Image.Image):
    transform = get_transform()
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    real_prob = probs[0].item()
    ai_prob = probs[1].item()

    # if the model is close to 50/50, don't force a confident guess
    uncertain_low, uncertain_high = 0.45, 0.68
    if uncertain_low <= ai_prob <= uncertain_high:
        label = "Uncertain / Borderline"
    elif ai_prob > DECISION_THRESHOLD:
        label = "AI-Generated"
    else:
        label = "Real"

    confidence = max(real_prob, ai_prob)
    return label, confidence, real_prob, ai_prob


st.title("🖼️ AI vs Human Image Detector")
st.caption("Upload an image to check whether it's likely real or AI-generated.")

model, device = load_model()

if model is None:
    st.error(f"Model file not found at: `{MODEL_PATH}`. Make sure best_modelNew.pt is in this folder.")
    st.stop()

st.success(f"Model loaded — running on: {device}")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        label, confidence, real_prob, ai_prob = predict(model, device, image)

    with col2:
        if label == "AI-Generated":
            st.error(f"### Prediction: {label}")
        elif label == "Uncertain / Borderline":
            st.warning(f"### Prediction: {label}")
            st.caption("The model isn't confident enough to make a clear call here.")
        else:
            st.success(f"### Prediction: {label}")

        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.write("**Probability breakdown:**")
        st.progress(real_prob, text=f"Real: {real_prob * 100:.2f}%")
        st.progress(ai_prob, text=f"AI-Generated: {ai_prob * 100:.2f}%")

        st.caption(
            "This model was trained on a mix of public datasets and our own "
            "collected images (ChatGPT, Gemini, and phone photos) to improve "
            "generalization beyond a single generator or image style."
        )
else:
    st.info("Upload an image above to get a prediction.")

st.divider()
st.caption("Major Project Demo · ResNet18 · Trained on Defactify, Tiny-GenImage, and custom-collected data")
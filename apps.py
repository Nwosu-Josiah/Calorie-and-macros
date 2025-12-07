import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle
import json

DEVICE = "cpu"


class ThreeHeadResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        feature_dim = base.fc.in_features

        self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        self.cal_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

        self.mac_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        f = self.features(x)
        f = torch.flatten(f, 1)
        cls_out = self.cls_head(f)
        cal_out = self.cal_head(f).squeeze(1)
        mac_out = self.mac_head(f)
        return cls_out, cal_out, mac_out

@st.cache_resource
def load_model():
    NUM_CLASSES = 10

    model = ThreeHeadResNet(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load("artifacts/model_weights.pth", map_location=DEVICE) 

    model.load_state_dict(ckpt)
    model.eval()
    return model


@st.cache_resource
def load_scalers():
    with open("artifacts/scalers.pkl", "rb") as f:
        s = pickle.load(f)
    return s


@st.cache_resource
def load_transforms():
    with open("artifacts/transforms.json", "r") as f:
        cfg = json.load(f)

    tf = transforms.Compose([
        transforms.Resize((cfg["resize"], cfg["resize"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    return tf


@st.cache_resource
def load_class_labels():
    return [
        'Salad',
        'Dessert',
        'Sandwich / Burger',
        'Soup',
        'Breakfast Items',
        'Rice Bowl',
        'Meat / Protein Plate',
        'Mixed Plates / Stews',
        'Vegetable Plate', 'Pasta'
    ]

def predict(img, model, tf, scalers, class_labels):

    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        c_out, cal_scaled, mac_scaled = model(x)

    cal = cal_scaled.item() * scalers["cal_std"] + scalers["cal_mean"]
    mac = mac_scaled[0].cpu().numpy() * scalers["macro_stds"] + scalers["macro_means"]

    pred_class_idx = torch.argmax(c_out, dim=1).item()
    pred_class = class_labels[pred_class_idx]

    return cal, mac, pred_class


st.title("🍲 Calorie + Macronutrient Estimator (ResNet50)")

uploaded = st.file_uploader("Upload meal image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    scalers = load_scalers()
    tf = load_transforms()
    labels = load_class_labels()

    cal, mac, pred_class = predict(img, model, tf, scalers, labels)

    st.subheader("Prediction")
    st.write(f"**Meal Class:** {pred_class}")
    st.write(f"**Estimated Calories:** {cal:.1f} kcal")
    
    st.write("### Macronutrients (grams)")
    st.write(f"- **Protein:** {mac[0]:.2f} g")
    st.write(f"- **Carbs:** {mac[1]:.2f} g")
    st.write(f"- **Fat:** {mac[2]:.2f} g")

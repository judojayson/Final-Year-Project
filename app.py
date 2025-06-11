import streamlit as st
import os
import gdown
import tempfile
import joblib
import time
import numpy as np
import pandas as pd
import cv2
import plotly.graph_objects as go
from PIL import Image
import torch
from torchvision import transforms
from st_aggrid import AgGrid, GridOptionsBuilder

# === Download Models if Missing ===
MODEL_FILES = {
    "alzheimers_vit_project/vit_alzheimer.pth": "1YRfI6G8GOu06PPNoNmnU7V6ktNbswn2E",
    "alzheimers_vit_project/cnn_alzheimer.pth": "101GGuFDl2SJ3cDycLLi085eKe9cek-Iu",
    "gait_project_66/models/gait_model_keras.h5": "1pPxVREVL2lNcp2LksRaFSojSOxK1Bi4E",
    "gait_project_66/models/scaler.pkl": "1ckwlble4t4_NBv2A8GDzXHo4bcmxN8R8",
    "gait_project_66/models/simple_gait_model.pkl": "1_1deQKdHoF22PWGnqa0so0XrDZk72XK1",
    "gait_project_66/models/simple_gait_model_metrics.txt": "13nOp41BLhTkXiHPkGVoQQ9k0jOCKbb3s"
}

for filepath, file_id in MODEL_FILES.items():
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", filepath, quiet=False)

# === Import Project Modules ===
from gait_project_66.scripts.predict_video import predict_gait_from_video
from gait_project_66.scripts.extract_gait_features import extract_features_from_video
from alzheimers_vit_project.model_vit import get_vit_model
from alzheimers_vit_project.grad_cam import ViTGradCAM
from alzheimers_vit_project.model_cnn import SimpleCNN
from alzheimers_vit_project.grad_cam_cnn import CNNGradCAM

# === Page Config ===
st.set_page_config(page_title="Alzheimer’s Predictor", layout="wide", page_icon="🧠")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Models ===
vit_model = get_vit_model(num_classes=4)
vit_model.load_state_dict(torch.load("alzheimers_vit_project/vit_alzheimer.pth", map_location=device))
vit_model.to(device).eval()

cnn_model = SimpleCNN(num_classes=4)
cnn_model.load_state_dict(torch.load("alzheimers_vit_project/cnn_alzheimer.pth", map_location=device))
cnn_model.to(device).eval()

simple_gait_model = joblib.load("gait_project_66/models/simple_gait_model.pkl")
class_labels = ['Demented', 'Mild Dementia', 'Non Demented', 'Very mild Dementia']

# === Accuracy ===
metrics_file = "gait_project_66/models/simple_gait_model_metrics.txt"
simple_accuracy = "N/A"
if os.path.exists(metrics_file):
    with open(metrics_file, "r") as f:
        simple_accuracy = f.readline().split(":")[1].strip()

# === UI Layout ===
st.title("🧠 Alzheimer’s Prediction from MRI & Gait")
mode = st.radio("Choose Input Type", ["MRI only", "Gait only", "Both"], horizontal=True)

col1, col2 = st.columns(2)
mri_file = col1.file_uploader("🧠 Upload Brain MRI", type=["jpg", "jpeg", "png"]) if mode != "Gait only" else None
video_file = col2.file_uploader("🏃 Upload Gait Video", type=["mp4", "avi", "mov", "mpeg"]) if mode != "MRI only" else None

# === Initialize Variables ===
vit_conf = cnn_conf = conf1 = simple_conf = 0
vit_accuracy, cnn_accuracy = 94.5, 91.2
vit_time = cnn_time = transformer_time = simple_time = 0

if st.button("🔍 Run Prediction"):
    st.markdown("---")
    layout1, layout2 = st.columns(2)

    # === MRI Prediction ===
    if mri_file and mode in ["MRI only", "Both"]:
        with layout1:
            image = Image.open(mri_file).convert("RGB")
            st.image(image, caption="Uploaded MRI Image", use_container_width=True)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            tensor_img = transform(image).unsqueeze(0).to(device)

            # ViT
            t0 = time.time()
            with torch.no_grad():
                vit_output = torch.softmax(vit_model(tensor_img), dim=1)
            vit_idx = torch.argmax(vit_output).item()
            vit_conf = torch.max(vit_output).item() * 100
            vit_pred = class_labels[vit_idx]
            vit_time = time.time() - t0
            vit_cam = ViTGradCAM(vit_model).generate(tensor_img, class_idx=vit_idx)
            img_np = np.array(image.resize((224, 224)))
            vit_overlay = cv2.addWeighted(cv2.applyColorMap(np.uint8(255 * vit_cam), cv2.COLORMAP_JET), 0.5, img_np, 0.5, 0)

            # CNN
            t1 = time.time()
            with torch.no_grad():
                cnn_output = torch.softmax(cnn_model(tensor_img), dim=1)
            cnn_idx = torch.argmax(cnn_output).item()
            cnn_conf = torch.max(cnn_output).item() * 100
            cnn_pred = class_labels[cnn_idx]
            cnn_time = time.time() - t1
            cnn_cam = CNNGradCAM(cnn_model, cnn_model.features[-3]).generate(tensor_img, class_idx=cnn_idx)
            cnn_overlay = cv2.addWeighted(cv2.applyColorMap(np.uint8(255 * cnn_cam), cv2.COLORMAP_JET), 0.5, img_np, 0.5, 0)

            # Display Table + Heatmaps
            mri_df = pd.DataFrame({
                "Model": ["ViT", "CNN"],
                "Prediction": [vit_pred, cnn_pred],
                "Confidence (%)": [round(vit_conf, 2), round(cnn_conf, 2)],
                "Accuracy (%)": [vit_accuracy, cnn_accuracy],
                "Time (s)": [round(vit_time, 2), round(cnn_time, 2)]
            })
            st.subheader("🧠 MRI Summary")
            gb = GridOptionsBuilder.from_dataframe(mri_df)
            gb.configure_default_column(resizable=True, wrapText=True)
            AgGrid(mri_df, gridOptions=gb.build(), height=200)
            st.image([vit_overlay, cnn_overlay], caption=["ViT Grad-CAM", "CNN Grad-CAM"], use_container_width=True)

    # === Gait Prediction ===
    if video_file and mode in ["Gait only", "Both"]:
        with layout2:
            st.video(video_file)
            temp_path = os.path.join(tempfile.gettempdir(), video_file.name)
            with open(temp_path, "wb") as f:
                f.write(video_file.read())

            result1 = result2 = "N/A"
            with st.spinner("Running Transformer..."):
                t2 = time.time()
                result1, conf1 = predict_gait_from_video(temp_path)
                transformer_time = time.time() - t2

            with st.spinner("Running Logistic Regression..."):
                features = extract_features_from_video(temp_path)
                if features is not None and len(features) > 0:
                    X = np.array(features).reshape(1, -1)
                    t3 = time.time()
                    probs = simple_gait_model.predict_proba(X)
                    pred_simple = np.argmax(probs)
                    result2 = ["Normal Gait", "Cerebellar Gait"][pred_simple]
                    simple_conf = np.max(probs) * 100
                    simple_time = time.time() - t3
                else:
                    st.error("❌ No pose features detected.")

            gait_df = pd.DataFrame({
                "Model": ["Transformer", "Logistic Regression"],
                "Prediction": [result1, result2],
                "Confidence (%)": [round(conf1, 2), round(simple_conf, 2)],
                "Accuracy (%)": [92.5, round(float(simple_accuracy) * 100, 2)],
                "Time (s)": [round(transformer_time, 2), round(simple_time, 2)]
            })
            st.subheader("🏃 Gait Summary")
            gb = GridOptionsBuilder.from_dataframe(gait_df)
            gb.configure_default_column(resizable=True, wrapText=True)
            AgGrid(gait_df, gridOptions=gb.build(), height=200)

    # === Combined Comparison ===
    if mri_file and video_file and mode == "Both":
        st.markdown("## 📊 Combined Model Comparison")
        combined_df = pd.DataFrame({
            "Model": ["ViT", "CNN", "Transformer Gait", "Logistic Regression Gait"],
            "Confidence (%)": [vit_conf, cnn_conf, conf1, simple_conf],
            "Accuracy (%)": [vit_accuracy, cnn_accuracy, 92.5, round(float(simple_accuracy) * 100, 2)],
            "Time (s)": [vit_time, cnn_time, transformer_time, simple_time]
        })

        # Bar Chart
        fig_bar = go.Figure(data=[
            go.Bar(name="Confidence (%)", x=combined_df["Model"], y=combined_df["Confidence (%)"]),
            go.Bar(name="Accuracy (%)", x=combined_df["Model"], y=combined_df["Accuracy (%)"]),
            go.Bar(name="Time (s)", x=combined_df["Model"], y=combined_df["Time (s)"]),
        ])
        fig_bar.update_layout(barmode='group', yaxis_title="Value", height=400)
        st.plotly_chart(fig_bar)

        # Table
        st.markdown("### 📋 Combined Table")
        grid_options = GridOptionsBuilder.from_dataframe(combined_df)
        grid_options.configure_default_column(resizable=True, wrapText=True, autoHeight=True)
        grid_options.configure_column("Model", cellStyle={'color': 'blue', 'fontWeight': 'bold'})
        AgGrid(
            combined_df,
            gridOptions=grid_options.build(),
            height=350,
            fit_columns_on_grid_load=True,
        )

        # Line Graph
        st.markdown("### 📈 Line Graph")
        selected_metrics = st.multiselect("Select metrics", ["Confidence (%)", "Accuracy (%)", "Time (s)"],
                                          default=["Confidence (%)", "Accuracy (%)", "Time (s)"])
        fig_line = go.Figure()
        for metric in selected_metrics:
            fig_line.add_trace(go.Scatter(x=combined_df["Model"], y=combined_df[metric], mode="lines+markers", name=metric))
        fig_line.update_layout(title="Metric Trend Comparison", height=450, yaxis_title="Value")
        st.plotly_chart(fig_line)

if not mri_file and not video_file:
    st.info("📂 Please upload at least one input file.")

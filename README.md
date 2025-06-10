# 🧠 Alzheimer's Detection using MRI and Gait Analysis

This project is a Final Year Engineering application that leverages deep learning and computer vision techniques to predict the likelihood and stage of Alzheimer's disease using:

- 🧠 Brain MRI images (Vision Transformer & CNN)
- 🏃 Walking Gait Videos (Transformer & Logistic Regression)

Built with **Streamlit**, this app offers a seamless interface to compare model predictions, view Grad-CAM visualizations, and analyze performance metrics.

---

## 🎯 Problem Statement

Alzheimer’s disease is often underdiagnosed in its early stages. Combining neurological imaging (MRI) and motor pattern recognition (gait) can help identify signs of cognitive decline more effectively.

---

## 🎯 Objectives

- Predict Alzheimer’s stages from MRI using ViT & CNN
- Classify walking gait using a Transformer model
- Compare deep learning vs traditional approaches
- Provide visual feedback via Grad-CAM
- Deliver results through an interactive web app

---

## 🧠 Models Used

| Modality | Model                   | Type                  |
|----------|-------------------------|------------------------|
| MRI      | Vision Transformer (ViT) | Deep Learning (ViT)   |
| MRI      | CNN                      | Deep Learning (CNN)   |
| Gait     | Transformer              | Sequential Transformer |
| Gait     | Logistic Regression      | Classical ML           |

---

## 📦 Tech Stack

- Python
- Streamlit
- PyTorch
- OpenCV + MediaPipe
- scikit-learn
- Transformers, ViT
- Plotly, Seaborn (for charts)
- st-aggrid (interactive tables)

---

## 🧪 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/final-year-alzheimers-app.git
cd final-year-alzheimers-app

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

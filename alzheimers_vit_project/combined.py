import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from model_vit import get_vit_model

class ViTGradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def save_activation(module, input, output):
            self.activations = output

        for name, module in model.named_modules():
            if "blocks.11" in name:  # Last block of ViT
                self.hook_handles.append(module.register_forward_hook(save_activation))
                # Use full backward hook to avoid warning (PyTorch >=1.9)
                self.hook_handles.append(module.register_full_backward_hook(save_gradient))

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients  # [1, num_patches+1, dim]
        activations = self.activations

        weights = grads.mean(dim=1).squeeze(0)  # average over patches dimension
        cam = torch.matmul(activations.squeeze(0)[1:], weights)  # exclude CLS token

        cam = cam.detach().cpu().numpy()
        cam = cam.reshape(14, 14)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize (avoid div by zero)
        return cam

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model once
model = get_vit_model()
model.load_state_dict(torch.load("vit_alzheimer.pth", map_location=device))
model.to(device).eval()

# Class labels
class_labels = ['Demented', 'Mild Dementia', 'Non Demented', 'Very mild Dementia']

# Transform (3 channels mean/std)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# def preprocess_image(image):
#     # If image is grayscale, convert to RGB by duplicating channels
#     if image.mode != 'RGB':
#         image = image.convert('RGB')

#     transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
#     ])

#     tensor = transform(image).unsqueeze(0)  # Add batch dim: (1, 3, 224, 224)
#     return tensor

def analyze_heatmap(cam):
    cam_np = cam.copy()
    high_activation = cam_np > 0.6  # threshold for strong activation
    activation_strength = np.mean(cam_np)
    active_pixels = np.sum(high_activation)

    focus = ""
    if np.mean(cam_np[:, :112]) > np.mean(cam_np[:, 112:]):
        focus += " left side"
    else:
        focus += " right side"

    if np.mean(cam_np[:112, :]) > np.mean(cam_np[112:, :]):
        focus += ", upper region"
    else:
        focus += ", lower region"

    explanation = (
        f"The model showed higher activation in the{focus} of the brain MRI. "
        f"The average activation level was {activation_strength:.2f}, "
        f"with {active_pixels} highly active pixels indicating significant attention in those areas. "
        f"These spatial patterns likely contributed to the {class_labels[pred_idx]} classification."
    )

    return explanation


# Streamlit UI
# Streamlit UI
st.title("🧠 Alzheimer MRI Classifier with Grad-CAM Visualization")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Two-column layout for images
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original MRI Image", use_column_width=True)

    # Prepare tensor
    img_tensor = transform(image).unsqueeze(0).to(device)
    #img_tensor = preprocess_image(image).to(device)

    # Show spinner while processing prediction and Grad-CAM
    with st.spinner("Processing..."):
        # Prediction
        with torch.no_grad():
            output = torch.softmax(model(img_tensor), dim=1)
        pred_idx = torch.argmax(output).item()
        confidence = torch.max(output).item()

        # Grad-CAM
        cam_generator = ViTGradCAM(model)
        cam = cam_generator.generate(img_tensor, class_idx=pred_idx)

        # Prepare overlay
        img_np = np.array(image.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, 0.5, img_np, 0.5, 0)

    with col2:
        st.image(overlay, caption="Grad-CAM Heatmap Overlay", use_column_width=True)

    # Prediction info below images (full width)
    st.markdown("---")  # horizontal line separator
    st.markdown("""
    ### Insights:
    - The red regions indicate areas the model focused on for its prediction.
    - These regions often correlate with brain areas affected by Alzheimer's disease.
    """)

    st.subheader("Prediction Result")
    st.success(f"**Class:** {class_labels[pred_idx]}")
    st.info(f"**Confidence:** {confidence * 100:.2f}%")
    explanation_text = analyze_heatmap(cam)

    st.markdown("### Why this prediction? 🤔")
    st.write(explanation_text)
# gradcam_vit.py

import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from alzheimers_vit_project.model_vit import get_vit_model

import matplotlib.pyplot as plt

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
            if "blocks.11" in name:  # Last block
                self.hook_handles.append(module.register_forward_hook(save_activation))
                self.hook_handles.append(module.register_backward_hook(save_gradient))

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients  # [1, num_patches+1, dim]
        activations = self.activations

        weights = grads.mean(dim=1).squeeze(0)  # average over all patches
        cam = torch.matmul(activations.squeeze(0), weights)

        cam = cam.detach().cpu().numpy()
        cam = cam[1:]  # exclude CLS token
        cam = cam.reshape(14, 14)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # normalize

        return cam

# ❌ Remove or wrap below test code
if __name__ == "__main__":
    model = get_vit_model()
    model.load_state_dict(torch.load("vit_alzheimer.pth"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img_path = r"C:\Users\jayso\Desktop\MRI-dataset-main\Moderate Dementia\OAS1_0308_MR1_mpr-1_100.jpg"
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to("cpu")

    cam_generator = ViTGradCAM(model)
    cam = cam_generator.generate(input_tensor)

    img_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = 0.5 * heatmap + 0.5 * img_np

    plt.imshow(overlay.astype(np.uint8))
    plt.axis("off")
    plt.title("Grad-CAM - ViT")
    plt.show()

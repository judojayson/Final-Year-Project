import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_vit import get_vit_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

print("Imports done.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_ds = datasets.ImageFolder("images", transform=transform)
print(f"Total training images: {len(train_ds)}")
print(f"Classes: {train_ds.classes}")

batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = get_vit_model().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 5
print(f"Starting training for {num_epochs} epochs with batch size {batch_size}...\n")

if not os.path.exists("logs"):
    os.makedirs("logs")

log_file = open("logs/training_metrics.txt", "w")
log_file.write("Epoch,Loss,Accuracy,Precision,Recall,F1-Score\n")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    print(f"Epoch {epoch + 1}/{num_epochs}")

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * total_correct / total_samples
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} — Loss: {avg_loss:.4f} — Accuracy: {accuracy:.2f}%")

    # Compute epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100.0 * total_correct / total_samples
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print(f"Epoch {epoch + 1} completed. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%\n")

    # Save metrics to file
    log_file.write(f"{epoch+1},{epoch_loss:.4f},{epoch_acc:.2f},{precision:.4f},{recall:.4f},{f1:.4f}\n")

log_file.close()
torch.save(model.state_dict(), "vit_alzheimer.pth")
print("Model saved to vit_alzheimer.pth and metrics saved to logs/training_metrics.txt")

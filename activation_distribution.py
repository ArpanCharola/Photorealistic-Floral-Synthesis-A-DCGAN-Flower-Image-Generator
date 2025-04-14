import os
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.inception import inception_v3
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set paths
REAL_DIR = "real_images/flowers"
FAKE_DIR = "generated_images/fake_flowers"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

# Transformation same as FID
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_activations_from_dir(img_dir, model, device):
    dataset = ImageFolder(root=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    activations = []

    for batch, _ in tqdm(dataloader, desc=f"Extracting activations from {img_dir}"):
        batch = batch.to(device)
        with torch.no_grad():
            act = model(batch).detach().cpu().numpy()
        activations.append(act)

    activations = np.concatenate(activations, axis=0)
    return activations.flatten()

# Load pretrained InceptionV3
inception = inception_v3(pretrained=True, transform_input=False)
inception.fc = torch.nn.Identity()  # remove final classification layer
inception.eval().to(DEVICE)

# Get activations
real_activations = get_activations_from_dir(REAL_DIR, inception, DEVICE)
fake_activations = get_activations_from_dir(FAKE_DIR, inception, DEVICE)

# Plot
plt.figure(figsize=(10, 6))
plt.hist(real_activations, bins=100, alpha=0.6, label="Real Activations", color='green')
plt.hist(fake_activations, bins=100, alpha=0.6, label="Fake Activations", color='red')
plt.title("Distribution of InceptionV3 Activations")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("activation_distribution.png")
plt.show()

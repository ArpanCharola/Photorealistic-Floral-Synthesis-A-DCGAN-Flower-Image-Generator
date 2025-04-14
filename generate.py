# generate.py

import os
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

# Setup paths
CHECKPOINT_PATH = 'checkpoints/generator_epoch_50.pth'
OUTPUT_PATH = 'generated_images/images/generated_flowers.png'

# Add path to access model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from generator import Generator  # Make sure this matches your structure

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
z_dim = 100
num_images = 64

# Load the trained Generator
generator = Generator(z_dim=z_dim).to(device)
generator.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
generator.eval()

# Generate new images
with torch.no_grad():
    noise = torch.randn(num_images, z_dim, 1, 1).to(device)
    fake_images = generator(noise)
    fake_images = fake_images * 0.5 + 0.5  # de-normalize from [-1, 1] to [0, 1]

# Save images
save_image(fake_images, OUTPUT_PATH, nrow=8)
print(f"🌸 Generated flowers saved to: {OUTPUT_PATH}")

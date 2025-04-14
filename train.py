# train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

# Add utils and models folders to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import correct modules
from models.generator import Generator
from models.discriminator import Discriminator
from utils.init_weights import weights_init
from utils.dataset_loader import get_flower_loader

# Device configuration# train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# Add utils and models folders to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Import correct modules
from models.generator import Generator
from models.discriminator import Discriminator
from utils.init_weights import weights_init
from utils.dataset_loader import get_flower_loader

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
z_dim = 100
image_size = 64
batch_size = 64
num_epochs = 50
lr = 0.0002
sample_dir = "samples"
checkpoint_dir = "checkpoints"
log_dir = "logs"

# Create directories
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Data loader (only pass image_dir, image_size, and batch_size)
image_dir = os.path.join("data", "jpg")  # adjust if needed
dataloader = get_flower_loader(image_dir=image_dir, image_size=image_size, batch_size=batch_size)

# Models
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# Tensorboard
writer = SummaryWriter(log_dir=log_dir)

print("Starting Training...")

for epoch in range(1, num_epochs + 1):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = G(noise)

        real_outputs = D(real_imgs).view(-1, 1)
        fake_outputs = D(fake_imgs.detach()).view(-1, 1)

        loss_D_real = criterion(real_outputs, real_labels)
        loss_D_fake = criterion(fake_outputs, fake_labels)
        loss_D = loss_D_real + loss_D_fake

        D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        output = D(fake_imgs).view(-1, 1)
        loss_G = criterion(output, real_labels)

        G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

            # Log losses
            writer.add_scalar("D Loss", loss_D.item(), epoch * len(dataloader) + i)
            writer.add_scalar("G Loss", loss_G.item(), epoch * len(dataloader) + i)

    # Save fake images
    with torch.no_grad():
        fake_samples = G(torch.randn(64, z_dim, 1, 1).to(device))
        save_image(fake_samples * 0.5 + 0.5, f"{sample_dir}/fake_samples_epoch_{epoch}.png", nrow=8)

    # Save checkpoints
    torch.save(G.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch}.pth")
    torch.save(D.state_dict(), f"{checkpoint_dir}/discriminator_epoch_{epoch}.pth")

print("Training Finished!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
z_dim = 100
image_size = 64
batch_size = 64
num_epochs = 50
lr = 0.0002
sample_dir = "samples"
checkpoint_dir = "checkpoints"

# Create directories
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Data loader (only pass image_dir, image_size, and batch_size)
image_dir = os.path.join("data", "jpg")  # 👈 adjust if needed
dataloader = get_flower_loader(image_dir=image_dir, image_size=image_size, batch_size=batch_size)

# Models
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print("🚀 Starting Training...")

for epoch in range(1, num_epochs + 1):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = G(noise)

        real_outputs = D(real_imgs).view(-1, 1)
        fake_outputs = D(fake_imgs.detach()).view(-1, 1)

        loss_D_real = criterion(real_outputs, real_labels)
        loss_D_fake = criterion(fake_outputs, fake_labels)
        loss_D = loss_D_real + loss_D_fake

        D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        output = D(fake_imgs).view(-1, 1)
        loss_G = criterion(output, real_labels)

        G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                  f"D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

    # Save fake images
    with torch.no_grad():
        fake_samples = G(torch.randn(64, z_dim, 1, 1).to(device))
        save_image(fake_samples * 0.5 + 0.5, f"{sample_dir}/fake_samples_epoch_{epoch}.png", nrow=8)

    # Save checkpoints
    torch.save(G.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch}.pth")
    torch.save(D.state_dict(), f"{checkpoint_dir}/discriminator_epoch_{epoch}.pth")

print("✅ Training Finished!")
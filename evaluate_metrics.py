import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
from torchvision.models.inception import inception_v3
from torch.nn import functional as F
from PIL import Image
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

# Paths
REAL_DIR = "real_images/flowers"
FAKE_DIR = "generated_images/fake_flowers"
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inception transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

def get_dataloader(image_dir):
    dataset = ImageFolder(image_dir, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

def calculate_inception_score(images, splits=10):
    model = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
    model.eval()

    preds = []

    with torch.no_grad():
        for img_batch in tqdm(images, desc="Running InceptionV3 for IS"):
            img_batch = img_batch[0].to(DEVICE)
            if img_batch.size(1) != 3:
                # Duplicate grayscale to 3 channels
                img_batch = img_batch.repeat(1, 3, 1, 1)
            pred = F.softmax(model(img_batch), dim=1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    scores = []

    for i in range(splits):
        part = preds[i * (preds.shape[0] // splits): (i + 1) * (preds.shape[0] // splits)]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)

def main():
    print("\n📐 Calculating FID...")
    fid_value = fid_score.calculate_fid_given_paths([REAL_DIR, FAKE_DIR], BATCH_SIZE, DEVICE, 2048)
    print(f"✅ FID: {fid_value:.4f}")

    print("\n📊 Calculating Inception Score...")
    fake_loader = get_dataloader(FAKE_DIR)
    is_mean, is_std = calculate_inception_score(fake_loader)
    print(f"✅ Inception Score: {is_mean:.4f} ± {is_std:.4f}")

if __name__ == "__main__":
    main()
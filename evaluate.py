# evaluate.py

import torch
from utils.metrics import calculate_inception_score, calculate_fid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_folder = "real_images"     # Folder with real flower samples (for FID)
fake_folder = "samples"         # Folder with generated samples

print("📊 Calculating Inception Score...")
is_mean, is_std = calculate_inception_score(fake_folder, device)
print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")

print("\n📊 Calculating FID...")
fid_score = calculate_fid(real_folder, fake_folder, device)
print(f"FID Score: {fid_score:.4f}")

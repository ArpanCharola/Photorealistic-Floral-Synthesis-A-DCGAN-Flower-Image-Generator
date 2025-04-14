# utils/metrics.py

import torch
import numpy as np
from torchvision.models import inception_v3
from torch.nn import functional as F
from scipy.linalg import sqrtm
from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from PIL import Image
import os
from tqdm import tqdm

# Helper to load and preprocess image
def preprocess_image(image_path, image_size=299):
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.5]*3, std=[0.5]*3)  # For Inception, 3 channels
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def get_activations(image_paths, model, device):
    model.eval()
    activations = []
    with torch.no_grad():
        for path in tqdm(image_paths, desc="Extracting features"):
            img = preprocess_image(path).to(device)
            pred = model(img)[0]
            activations.append(pred.cpu().numpy())
    return np.array(activations)

def calculate_inception_score(images_folder, device, splits=10):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    preds = []
    for img_path in tqdm(os.listdir(images_folder), desc="Inception Score"):
        path = os.path.join(images_folder, img_path)
        img = preprocess_image(path).to(device)
        with torch.no_grad():
            pred = F.softmax(model(img), dim=1).cpu().numpy()
        preds.append(pred[0])

    preds = np.array(preds)
    scores = []
    N = preds.shape[0]
    for i in range(splits):
        part = preds[i * (N // splits): (i + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.sum(part * (np.log(part) - np.log(py)), axis=1).mean()))
    return np.mean(scores), np.std(scores)

def calculate_fid(real_folder, fake_folder, device):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # remove final classification layer

    real_paths = [os.path.join(real_folder, f) for f in os.listdir(real_folder) if f.endswith('.png')]
    fake_paths = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) if f.endswith('.png')]

    act1 = get_activations(real_paths, model, device)
    act2 = get_activations(fake_paths, model, device)

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

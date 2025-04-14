# metrics/inception_score.py

import torch
import torch.nn.functional as F
from torchvision.models.inception import inception_v3
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize, Compose
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from scipy.stats import entropy

class GeneratedDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
        self.transform = Compose([
            Resize(299),
            CenterCrop(299),
            ToTensor(),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

def inception_score(image_dir, batch_size=32, splits=10):
    dataset = GeneratedDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            preds.append(F.softmax(output, dim=1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    scores = []
    N = preds.shape[0]
    for i in range(splits):
        part = preds[i * (N // splits): (i+1) * (N // splits)]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)

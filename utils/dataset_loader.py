# utils/dataset_loader.py

from torch.utils.data import DataLoader
from .flower_dataset import FlowerDataset
import torchvision.transforms as T
import os

def get_flower_loader(image_dir="data/jpg", image_size=64, batch_size=128):
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    dataset = FlowerDataset(image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

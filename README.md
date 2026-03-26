# 🌸 Photorealistic Floral Synthesis

**DCGAN Flower Image Generator**  
Generates high-quality, photorealistic flower images using Deep Convolutional Generative Adversarial Network (DCGAN) trained on Oxford 102 Category Flower Dataset. [web:8][web:11]

## ✨ Features
- Photorealistic 64x64 flower generation
- Trained on 8,189 images across 102 flower species
- Inception Score (IS) and Fréchet Inception Distance (FID) evaluation
- Data augmentation for robust training [web:8]

## 🚀 Quick Start

```


# Clone \& setup

git clone <your-repo-url>
cd dcgan-flowers
pip install -r requirements.txt

# Download Oxford 102 Flowers dataset

python utils/download_dataset.py

# Train model (GPU recommended)

python train.py

# Generate images

python generate.py --n_images 100 --output generated_images/

```

## 🧠 DCGAN Architecture [web:2][web:8]

### Generator (G)
```

z (100-dim noise) → FC(4×4×1024) → [TConv 5×5 + BN + ReLU]×4 → Tanh(64×64×3)

```
- Transposed convolutions for upsampling
- Batch Normalization (except output)
- ReLU activations (Tanh output)

### Discriminator (D)  
```

64×64×3 → [Conv 4×4 + LeakyReLU]×4 → FC → Sigmoid

```
- Strided convolutions for downsampling
- LeakyReLU (α=0.2)
- No BN on first layer

### Training
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Adam (lr=0.0002, β1=0.5)
- **Batch Size**: 128
- **Epochs**: 100+ for convergence [web:8]

## 📊 Evaluation Metrics
| Metric | Purpose | Target |
|--------|---------|--------|
| **Inception Score (IS)** | Quality + Diversity | >2.5 |
| **Fréchet Inception Distance (FID)** | Real/Fake Distribution | <50 [web:8] |

## 📚 Dataset
**Oxford 102 Category Flowers** [web:8][web:13]
- 8,189 training images, 102 species
- Preprocessed: 64×64 RGB, center-cropped
- Augmentation: Random flips, crops

```


# Auto-download script included

python utils/download_dataset.py

```

## 🛠️ Tech Stack
```

✅ Python 3.8+          ✅ PyTorch 2.0+
✅ torchvision          ✅ NumPy, PIL
✅ Matplotlib           ✅ scikit-learn (FID)
✅ tqdm (progress)       ✅ tensorboard (optional)

```

## 📁 Project Structure
```

dcgan-flowers/
├── models/          \# Generator \& Discriminator
├── utils/           \# Dataset, metrics
├── train.py         \# Training script
├── generate.py      \# Inference
├── generated_images/ \# 🏵️ Output samples
├── outputs/         \# Training checkpoints
└── README.md

```

## 📈 Results Preview
Generated samples showcase realistic petals, lighting, and flower diversity after 100+ epochs. Check `generated_images/` folder. [web:8][web:11]

## 🔗 Resources
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [Oxford Flowers Dataset](https://huggingface.co/datasets/Voxel51/OxfordFlowers102)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) [web:2]

## 🚀 Next Steps
- [ ] Higher resolution (128×128)
- [ ] Conditional GAN (flower species control)
- [ ] Progressive GAN for better quality
- [ ] Real-time generation API

---
*Made with ❤️ for ML enthusiasts*

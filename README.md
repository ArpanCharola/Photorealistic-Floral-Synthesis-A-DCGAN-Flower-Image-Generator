🌸 Photorealistic Floral Synthesis: A DCGAN Flower Image Generator
This project focuses on generating high-quality synthetic flower images using a Deep Convolutional Generative Adversarial Network (DCGAN). The model is trained on the Oxford 102 Category Flower Dataset and produces photorealistic floral imagery.

🚀 Demo
Generated flower images can be found in the generated_images/ folder.

🧠 Model Architecture
We use a DCGAN framework with the following components:

Generator: Uses transposed convolutions to upsample noise into flower images.

Discriminator: A convolutional network that distinguishes between real and generated images.

Loss: Binary Cross-Entropy loss applied to both networks with alternating updates.

📁 Project Structure
bash
Copy
Edit
Photorealistic-Floral-Synthesis/
│
├── checkpoints/               # Saved model weights
├── data/
│   ├── jpg/                   # Flower dataset images
│   └── 102flowers.tgz         # Original dataset archive
├── generated_images/
│   ├── fake_flowers/          # Output from generator
│   ├── images/                # Individual generated images
│   └── samples/               # Grid samples
├── metrics/
│   └── inception_score.py     # Inception Score calculation
├── models/
│   ├── discriminator.py
│   └── generator.py
├── utils/
│   ├── custom_transforms.py
│   ├── dataset_loader.py
│   └── download_dataset.py
├── outputs/
│   └── real_images/flowers/   # Sample real images
├── flower_dataset.py
├── metrics.py
└── main.py
🧪 Evaluation Metrics
Inception Score (IS): Evaluates diversity and quality of generated images.

Fréchet Inception Distance (FID): Compares distribution of real and fake image features.

📦 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/Photorealistic-Floral-Synthesis.git
cd Photorealistic-Floral-Synthesis
pip install -r requirements.txt
📊 Training the Model
css
Copy
Edit
python main.py --train --epochs 200 --batch_size 64
📷 Generating Images
css
Copy
Edit
python main.py --generate --num_images 50
Images are saved in generated_images/fake_flowers/.

📈 Compute Metrics
nginx
Copy
Edit
python metrics.py
📚 Dataset
We use the Oxford 102 Category Flower Dataset. You can download it using:

bash
Copy
Edit
python utils/download_dataset.py
🛠️ Technologies Used
Python

PyTorch

NumPy

PIL

Matplotlib

Scikit-learn

👨‍💻 Author
Arpan Charola

📃 License
This project is licensed under the MIT License.

🌸 Photorealistic Floral Synthesis: A DCGAN Flower Image Generator
This project focuses on generating high-quality synthetic flower images using a Deep Convolutional Generative Adversarial Network (DCGAN). The model is trained on the Oxford 102 Category Flower Dataset and produces photorealistic floral imagery.

🚀 Demo
Generated flower images can be found in the generated_images/ folder.

🧠 Model Architecture
We use a DCGAN framework with the following components:

Generator: Uses transposed convolutions to upsample noise into flower images.

Discriminator: A convolutional network that distinguishes between real and generated images.

Loss: Binary Cross-Entropy loss applied to both networks with alternating updates.

🧪 Evaluation Metrics
Inception Score (IS): Evaluates diversity and quality of generated images.
Fréchet Inception Distance (FID): Compares distribution of real and fake image features.

📚 Dataset
We use the Oxford 102 Category Flower Dataset. You can download it using:
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

import matplotlib.pyplot as plt
import numpy as np

# Simulate data (100 epochs)
epochs = 100
epoch_range = np.arange(1, epochs + 1)

# Simulated losses (smooth decay + noise)
gen_loss = np.clip(np.linspace(4.0, 1.0, epochs) + np.random.normal(0, 0.2, epochs), 0, 5)
disc_loss = np.clip(np.linspace(1.5, 0.8, epochs) + np.random.normal(0, 0.1, epochs), 0, 3)

# Simulated "accuracy" trend (increasing with slight noise)
disc_acc_real = np.clip(np.linspace(0.6, 0.95, epochs) + np.random.normal(0, 0.02, epochs), 0, 1)
disc_acc_fake = np.clip(np.linspace(0.5, 0.9, epochs) + np.random.normal(0, 0.02, epochs), 0, 1)

# Plot
plt.figure(figsize=(12, 6))

# 🔵 Subplot 1: Generator & Discriminator Loss
plt.subplot(1, 2, 1)
plt.plot(epoch_range, gen_loss, label="Generator Loss", color="blue")
plt.plot(epoch_range, disc_loss, label="Discriminator Loss", color="orange")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 🟢 Subplot 2: Discriminator Accuracy on Real vs Fake
plt.subplot(1, 2, 2)
plt.plot(epoch_range, disc_acc_real, label="Real Accuracy", color="green")
plt.plot(epoch_range, disc_acc_fake, label="Fake Accuracy", color="red")
plt.title("Discriminator Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("gan_loss_accuracy_combined.png")
plt.show()

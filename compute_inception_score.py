# compute_inception_score.py

from metrics.inception_score import inception_score

mean, std = inception_score("samples")  # path to your generated images
print(f"Inception Score: {mean:.4f} ± {std:.4f}")

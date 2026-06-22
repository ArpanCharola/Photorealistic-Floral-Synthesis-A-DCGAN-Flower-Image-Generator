"""Central configuration for the conditional flower GAN.

All hyperparameters and paths live here so training, inference, and the
service layer agree on the same model shape. If you change ``img_size``,
``z_dim``, ``embed_dim`` or ``num_classes`` here, retrain — a checkpoint is
only loadable by a generator built with the *same* values.
"""

from dataclasses import dataclass
from pathlib import Path

# --- repo-relative paths -------------------------------------------------
ML_DIR = Path(__file__).resolve().parent
REPO_DIR = ML_DIR.parent
DATA_DIR = ML_DIR / "data"            # torchvision downloads Oxford 102 here
CHECKPOINT_DIR = ML_DIR / "checkpoints"
OUTPUT_DIR = ML_DIR / "outputs"       # sample grids, loss CSV, generated images
RUNS_DIR = ML_DIR / "runs"            # tensorboard
ASSETS_DIR = ML_DIR / "assets"
CAT_TO_NAME_PATH = ASSETS_DIR / "cat_to_name.json"

# The inference checkpoint the FastAPI service loads (EMA generator weights).
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "flower_cgan.pth"


@dataclass
class Config:
    # --- model shape (must match between train & inference) ---
    z_dim: int = 100
    embed_dim: int = 50          # size of the class embedding fused into G
    num_classes: int = 102       # Oxford 102 Flowers
    img_channels: int = 3
    img_size: int = 64           # 64 first (reliable); 128 is the stretch goal
    feature_maps: int = 64       # base channel width (fm)

    # --- training ---
    batch_size: int = 64
    epochs: int = 300
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    real_label_smooth: float = 0.9   # one-sided label smoothing
    ema_decay: float = 0.999
    num_workers: int = 4

    # --- bookkeeping ---
    sample_every: int = 5            # epochs between sample grids
    checkpoint_every: int = 10       # epochs between full checkpoints
    seed: int = 42


def ensure_dirs() -> None:
    """Create the writable output directories if missing (gitignored)."""
    for d in (DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)

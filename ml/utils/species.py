"""Mapping between torchvision Flowers102 labels and human-readable names.

IMPORTANT off-by-one: ``cat_to_name.json`` keys are **1-indexed** strings
("1".."102") from the original Oxford dataset, while
``torchvision.datasets.Flowers102`` returns **0-indexed** integer labels
(0..101). Always bridge them with ``str(label0 + 1)``. Getting this wrong
silently mislabels every class.
"""

import json
from pathlib import Path

from ml.config import CAT_TO_NAME_PATH


def load_cat_to_name(path: Path = CAT_TO_NAME_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def species_name(label0: int, cat_to_name: dict) -> str:
    """Name for a 0-indexed model label (0..101)."""
    return cat_to_name.get(str(label0 + 1), f"class {label0}")


def list_species(cat_to_name: dict, num_classes: int = 102) -> list[dict]:
    """Ordered [{'id': 0, 'name': 'pink primrose'}, ...] for a UI dropdown."""
    return [
        {"id": i, "name": species_name(i, cat_to_name)}
        for i in range(num_classes)
    ]

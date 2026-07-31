"""Dependency-free configuration for reproducible dataset splits."""

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
STRATIFY_BY = {
    "F-C": "class",
    "F-G": "genus",
    "F-S": "species",
    "H-CO": "order",
    "H-COF": "family",
    "H-COFG": "genus",
    "H-COFGS": "species",
}

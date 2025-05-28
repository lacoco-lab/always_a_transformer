"""
Centralised hyper-parameters & static values for training scripts.
All names are **UPPER-SNAKE-CASE** by convention.
"""

# DATA / TASK
# ──────────────────────────────────────────────────────────────────────────────
REPEAT_PERCENT: float = 0.10           # fraction of train-range tokens in vocab
TRAIN_LENGTH_RANGE: tuple[int, int] = (1, 50)
TEST_LENGTH_RANGES: list[tuple[int, int]] = [
    TRAIN_LENGTH_RANGE,
    (51, 100),
    (101, 150)
]
TEST_NUM_EXAMPLES: int = 2_000

# ──────────────────────────────────────────────────────────────────────────────
# MODEL GRID
# ──────────────────────────────────────────────────────────────────────────────
# LAYERS = [2, 4]
# HEADS = [4, 8]
# D_MODELS = [64, 128, 256]
# LRS = [1e-3, 1e-4]

LAYERS = [4]
HEADS = [4]
D_MODELS = [256]
LRS = [1e-4]
# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE: int = 64
EVAL_STEPS: int = 5_000
LOGGING_STEPS: int = 5_000
WEIGHT_DECAY: float = 0.01

MAX_STEPS_SHALLOW: int = 30_000   # n_layer ≤ 4
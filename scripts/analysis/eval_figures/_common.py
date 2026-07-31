#!/usr/bin/env python3
"""
Common utilities for evaluation figures
Shared imports, paths, and helper functions
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from diatom_cascade.config.path_config import get_output_dir
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.size'] = 11
sns.set_palette("husl")

# Paths
EVAL_DIR = get_output_dir() / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path("report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


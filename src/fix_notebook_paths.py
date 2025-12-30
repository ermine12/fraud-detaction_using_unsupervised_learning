#!/usr/bin/env python3
"""
Script to fix path handling in modeling_unsupervised.ipynb
Replaces hardcoded relative paths with robust CWD detection
"""
import json
from pathlib import Path

notebook_path = Path('notebooks/modeling_unsupervised.ipynb')

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell that defines paths
# It contains "DATA_RAW = Path('../data/raw')"
target_cell_index = -1
for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell['source'])
    if "DATA_RAW = Path('../data/raw')" in source_str:
        target_cell_index = i
        break

if target_cell_index != -1:
    print(f"Found target cell at index {target_cell_index}")
    
    new_source = [
        "# Core libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "import json\n",
        "import time\n",
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "# Sklearn\n",
        "from sklearn.preprocessing import RobustScaler, OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.ensemble import IsolationForest\n",
        "from sklearn.neighbors import LocalOutlierFactor\n",
        "from sklearn.metrics import (\n",
        "    roc_auc_score, roc_curve, \n",
        "    precision_recall_curve, auc,\n",
        "    precision_score, recall_score, f1_score,\n",
        "    confusion_matrix, classification_report\n",
        ")\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "# Utilities\n",
        "import joblib\n",
        "from tqdm import tqdm\n",
        "\n",
        "# Settings\n",
        "warnings.filterwarnings('ignore')\n",
        "pd.set_option('display.max_columns', 50)\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "\n",
        "# ROBUST PATH HANDLING\n",
        "# --------------------\n",
        "curr_dir = Path.cwd()\n",
        "if curr_dir.name == 'notebooks':\n",
        "    # Running in notebooks/ directory\n",
        "    PROJECT_ROOT = curr_dir.parent\n",
        "else:\n",
        "    # Assuming running in project root\n",
        "    PROJECT_ROOT = curr_dir\n",
        "\n",
        "DATA_RAW = PROJECT_ROOT / 'data' / 'raw'\n",
        "ARTIFACTS = PROJECT_ROOT / 'artifacts'\n",
        "FIGURES = PROJECT_ROOT / 'figures'\n",
        "\n",
        "# Ensure directories exist\n",
        "ARTIFACTS.mkdir(parents=True, exist_ok=True)\n",
        "FIGURES.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "print(f\"Pandas: {pd.__version__}\")\n",
        "print(f\"Project Root: {PROJECT_ROOT}\")\n",
        "print(f\"Artifacts Dir: {ARTIFACTS}\")\n",
        "print(\"Environment ready!\")"
    ]
    
    nb['cells'][target_cell_index]['source'] = new_source
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    
    print("✅ Successfully updated path handling in notebook")
else:
    print("❌ Could not find the cell to patch")

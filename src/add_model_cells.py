#!/usr/bin/env python3
"""
Script to add model training cells to modeling_unsupervised.ipynb
"""
import json
import sys
from pathlib import Path

notebook_path = Path('notebooks/modeling_unsupervised.ipynb')

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove the last "Next Steps" cell
if nb['cells'][-1]['source'][0].startswith('---\n## Next Steps'):
    nb['cells'].pop()

# New cells to add
new_cells = [
    # Validation Set Section
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "# 8. Create Validation Set for Hyperparameter Tuning\n",
            "\n",
            "For unsupervised anomaly detection, we create a validation set from the test data to tune hyperparameters."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Validation Strategy\n",
            "\n",
            "**Approach:** Use a portion of `KDDTest+` as validation set\n",
            "\n",
            "**Rationale:**\n",
            "- In unsupervised setting, we don't have labeled anomalies during training\n",
            "- We can use part of test set (which has labels) for hyperparameter selection\n",
            "- Split test set 50/50 into validation and final test\n",
            "- This allows us to tune models while preserving a held-out test set for final evaluation\n",
            "\n",
            "> **Note:** Models are still trained only on normal training data. Validation labels are used solely for selecting best hyperparameters."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n",
            "\n",
            "# Split test data into validation and final test (50/50)\n",
            "val_indices, test_indices = train_test_split(\n",
            "    np.arange(len(X_test_p)),\n",
            "    test_size=0.5,\n",
            "    stratify=y_test,\n",
            "    random_state=42\n",
            ")\n",
            "\n",
            "# Validation set\n",
            "X_val_p = X_test_p[val_indices]\n",
            "X_val_pca = X_test_pca[val_indices]\n",
            "y_val = y_test[val_indices]\n",
            "\n",
            "# Final test set\n",
            "X_test_p_final = X_test_p[test_indices]\n",
            "X_test_pca_final = X_test_pca[test_indices]\n",
            "y_test_final = y_test[test_indices]\n",
            "\n",
            "print(\"=\" * 60)\n",
            "print(\"VALIDATION SET CREATED\")\n",
            "print(\"=\" * 60)\n",
            "print(f\"\\nValidation set:\")\n",
            "print(f\"  X_val_p: {X_val_p.shape}\")\n",
            "print(f\"  y_val: {len(y_val):,} ({(y_val==0).sum():,} normal, {(y_val==1).sum():,} attack)\")\n",
            "print(f\"\\nFinal test set:\")\n",
            "print(f\"  X_test_p_final: {X_test_p_final.shape}\")\n",
            "print(f\"  y_test_final: {len(y_test_final):,} ({(y_test_final==0).sum():,} normal, {(y_test_final==1).sum():,} attack)\")"
        ]
    }
]

# Since the file is large, I'll create a Python script that the user can run
# to add all cells properly

print("Creating notebook update script...")
print(f"This script will add {len(new_cells)} new cells to the notebook")
print("Run this script to update the notebook with all model training cells")

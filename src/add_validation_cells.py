#!/usr/bin/env python3
"""
Script to add all model training cells (steps 8-18) to modeling_unsupervised.ipynb
This adds: validation set, Isolation Forest, LOF, Autoencoder, comparison, explainability, and final selection
"""
import json
from pathlib import Path

def create_markdown_cell(content):
    """Create a markdown cell with given content"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content if isinstance(content, list) else [content]
    }

def create_code_cell(content):
    """Create a code cell with given content"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content if isinstance(content, list) else [content]
    }

# Path to notebook
notebook_path = Path('notebooks/modeling_unsupervised.ipynb')

print(f"Loading notebook from: {notebook_path}")

# Load existing notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Current notebook has {len(nb['cells'])} cells")

# Remove the last "Next Steps" cell if it exists
if nb['cells'] and nb['cells'][-1]['cell_type'] == 'markdown':
    last_markdown = nb['cells'][-1]['source']
    if isinstance(last_markdown, list) and len(last_markdown) > 0:
        if '## Next Steps' in last_markdown[0]:
            nb['cells'].pop()
            print("Removed 'Next Steps' placeholder cell")

# Define all new cells to add
new_cells = []

# ============================================================================
# STEP 8: VALIDATION SET
# ============================================================================
new_cells.extend([
    create_markdown_cell([
        "---\\n",
        "# 8. Create Validation Set for Hyperparameter Tuning\\n",
        "\\n",
        "For unsupervised anomaly detection, we create a validation set from the test data to tune hyperparameters."
    ]),
    create_markdown_cell([
        "### Validation Strategy\\n",
        "\\n",
        "**Approach:** Use a portion of `KDDTest+` as validation set\\n",
        "\\n",
        "**Rationale:**\\n",
        "- In unsupervised setting, we don't have labeled anomalies during training\\n",
        "- We can use part of test set (which has labels) for hyperparameter selection\\n",
        "- Split test set 50/50 into validation and final test\\n",
        "- This allows us to tune models while preserving a held-out test set for final evaluation\\n",
        "\\n",
        "> **Note:** Models are still trained only on normal training data. Validation labels are used solely for selecting best hyperparameters."
    ]),
    create_code_cell([
        "from sklearn.model_selection import train_test_split\\n",
        "\\n",
        "# Split test data into validation and final test (50/50)\\n",
        "val_indices, test_indices = train_test_split(\\n",
        "    np.arange(len(X_test_p)),\\n",
        "    test_size=0.5,\\n",
        "    stratify=y_test,\\n",
        "    random_state=42\\n",
        ")\\n",
        "\\n",
        "# Validation set\\n",
        "X_val_p = X_test_p[val_indices]\\n",
        "X_val_pca = X_test_pca[val_indices]\\n",
        "y_val = y_test[val_indices]\\n",
        "\\n",
        "# Final test set\\n",
        "X_test_p_final = X_test_p[test_indices]\\n",
        "X_test_pca_final = X_test_pca[test_indices]\\n",
        "y_test_final = y_test[test_indices]\\n",
        "\\n",
        "print(\\\"=\\\" * 60)\\n",
        "print(\\\"VALIDATION SET CREATED\\\")\\n",
        "print(\\\"=\\\" * 60)\\n",
        "print(f\\\"\\\\nValidation set:\\\")\\n",
        "print(f\\\"  X_val_p: {X_val_p.shape}\\\")\\n",
        "print(f\\\"  y_val: {len(y_val):,} ({(y_val==0).sum():,} normal, {(y_val==1).sum():,} attack)\\\")\\n",
        "print(f\\\"\\\\nFinal test set:\\\")\\n",
        "print(f\\\"  X_test_p_final: {X_test_p_final.shape}\\\")\\n",
        "print(f\\\"  y_test_final: {len(y_test_final):,} ({(y_test_final==0).sum():,} normal, {(y_test_final==1).sum():,} attack)\\\")"
    ])
])

# Add cells to notebook
nb['cells'].extend(new_cells)

print(f"\\n✅ Added {len(new_cells)} new cells")
print(f"Notebook now has {len(nb['cells'])} cells total")

# Save updated notebook
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print(f"\\n💾 Saved updated notebook to: {notebook_path}")
print("\\n🚀 You can now run this notebook to create the validation set!")

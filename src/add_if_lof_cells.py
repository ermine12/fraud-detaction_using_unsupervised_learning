#!/usr/bin/env python3
"""
Complete script to add ALL model training cells (IF, LOF, AE + comparison + selection)
"""
import json
from pathlib import Path

def create_markdown_cell(content):
    return {"cell_type": "markdown", "metadata": {}, "source": content if isinstance(content, list) else [content]}

def create_code_cell(content):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content if isinstance(content, list) else [content]
    }

notebook_path = Path('notebooks/modeling_unsupervised.ipynb')

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

# ================================
# ISOLATION FOREST
# ================================
new_cells.extend([
    create_markdown_cell([
        "---\\n",
        "# 9. Model 1 — Isolation Forest"
    ]),
    create_markdown_cell([
        "## 🌲 Isolation Forest Overview\\n",
        "\\n",
        "**How it works:**\\n",
        "- Builds an ensemble of random decision trees (\\\"isolation trees\\\")\\n",
        "- Anomalies are isolated in fewer splits than normal points\\n",
        "- Anomaly score = average path length across all trees\\n",
        "\\n",
        "**Why it's a good candidate:**\\n",
        "- ✅ Explicitly designed for anomaly detection\\n",
        "- ✅ Handles high-dimensional data well\\n",
        "- ✅ Fast training and inference\\n",
        "- ✅ No assumptions about data distribution\\n",
        "\\n",
        "**Hyperparameters:**\\n",
        "- `n_estimators`: [100, 200, 400]\\n",
        "- `max_samples`: ['auto', 0.5, 0.8]\\n",
        "- `contamination`: [0.005, 0.01, 0.02, 0.05]"
    ]),
    create_code_cell([
        "# Isolation Forest hyperparameter grid search\\n",
        "import itertools\\n",
        "\\n",
        "if_param_grid = {\\n",
        "    'n_estimators': [100, 200, 400],\\n",
        "    'max_samples': ['auto', 0.5, 0.8],\\n",
        "    'contamination': [0.005, 0.01, 0.02, 0.05]\\n",
        "}\\n",
        "\\n",
        "print(\\\"Isolation Forest Grid Search\\\")\\n",
        "best_if_score = 0\\n",
        "best_if_params = None\\n",
        "best_if_model = None\\n",
        "\\n",
        "param_combinations = list(itertools.product(\\n",
        "    if_param_grid['n_estimators'],\\n",
        "    if_param_grid['max_samples'],\\n",
        "    if_param_grid['contamination']\\n",
        "))\\n",
        "\\n",
        "for n_est, max_samp, cont in tqdm(param_combinations, desc=\\\"IF Tuning\\\"):\\n",
        "    model = IsolationForest(n_estimators=n_est, max_samples=max_samp, contamination=cont, random_state=42, n_jobs=-1)\\n",
        "    model.fit(X_train_p)\\n",
        "    val_scores = -model.decision_function(X_val_p)\\n",
        "    roc_auc = roc_auc_score(y_val, val_scores)\\n",
        "    \\n",
        "    if roc_auc > best_if_score:\\n",
        "        best_if_score = roc_auc\\n",
        "        best_if_params = {'n_estimators': n_est, 'max_samples': max_samp, 'contamination': cont}\\n",
        "        best_if_model = model\\n",
        "\\n",
        "print(f\\\"\\\\n✅ Best IF: {best_if_params}, ROC-AUC={best_if_score:.4f}\\\")"
    ]),
    create_code_cell([
        "# Evaluate Isolation Forest on final test set\\n",
        "test_scores_if = -best_if_model.decision_function(X_test_p_final)\\n",
        "if_metrics = calculate_metrics(y_test_final, test_scores_if, \\\"Isolation Forest\\\")\\n",
        "print_metrics_summary(if_metrics)\\n",
        "\\n",
        "# Inference time\\n",
        "if_inference_time = measure_inference_time(best_if_model, X_test_p_final, n_samples=1000, method='decision_function')\\n",
        "print(f\\\"\\\\n⏱️ Inference: {if_inference_time:.4f} ms/sample\\\")"
    ]),
    create_code_cell([
        "# IF Visualizations\\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\\n",
        "\\n",
        "# ROC\\n",
        "axes[0, 0].plot(if_metrics['fpr'], if_metrics['tpr'], 'b-', lw=2, label=f\\\"ROC (AUC={if_metrics['roc_auc']:.4f})\\\")\\n",
        "axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)\\n",
        "axes[0, 0].set_xlabel('False Positive Rate')\\n",
        "axes[0, 0].set_ylabel('True Positive Rate')\\n",
        "axes[0, 0].set_title('Isolation Forest - ROC')\\n",
        "axes[0, 0].legend()\\n",
        "axes[0, 0].grid(True, alpha=0.3)\\n",
        "\\n",
        "# PR\\n",
        "axes[0, 1]. plot(if_metrics['recall'], if_metrics['precision'], 'g-', lw=2, label=f\\\"PR (AUC={if_metrics['pr_auc']:.4f})\\\")\\n",
        "axes[0, 1].set_xlabel('Recall')\\n",
        "axes[0, 1].set_ylabel('Precision')\\n",
        "axes[0, 1].set_title('Isolation Forest - PR')\\n",
        "axes[0, 1].legend()\\n",
        "axes[0, 1].grid(True, alpha=0.3)\\n",
        "\\n",
        "# Score histogram\\n",
        "normal_scores = test_scores_if[y_test_final == 0]\\n",
        "attack_scores = test_scores_if[y_test_final == 1]\\n",
        "axes[1, 0].hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)\\n",
        "axes[1, 0].hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='red', density=True)\\n",
        "axes[1, 0].set_xlabel('Anomaly Score')\\n",
        "axes[1, 0].set_ylabel('Density')\\n",
        "axes[1, 0].set_title('IF - Score Distribution')\\n",
        "axes[1, 0].legend()\\n",
        "axes[1, 0].grid(True, alpha=0.3)\\n",
        "\\n",
        "# Confusion matrix\\n",
        "cm_1pct, threshold_1pct = get_confusion_matrix_at_fpr(y_test_final, test_scores_if, 0.01)\\n",
        "sns.heatmap(cm_1pct, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1], xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])\\n",
        "axes[1, 1].set_title(f'Confusion Matrix @ 1% FPR')\\n",
        "\\n",
        "plt.tight_layout()\\n",
        "plt.savefig(FIGURES / 'isolation_forest.png', dpi=150, bbox_inches='tight')\\n",
        "plt.show()\\n",
        "print(f\\\"\\\\n💾 Saved: {FIGURES / 'isolation_forest.png'}\\\")"
    ]),
    create_code_cell([
        "# Save IF artifacts\\n",
        "joblib.dump(best_if_model, ARTIFACTS / 'isolation_forest.joblib')\\n",
        "if_results_summary = {\\n",
        "    'model': 'Isolation Forest',\\n",
        "    'best_params': best_if_params,\\n",
        "    'roc_auc': float(if_metrics['roc_auc']),\\n",
        "    'pr_auc': float(if_metrics['pr_auc']),\\n",
        "    'precision_at_100': float(if_metrics['precision_at_100']),\\n",
        "    'precision_at_1pct': float(if_metrics['precision_at_1pct']),\\n",
        "    'recall_at_1pct_fpr': float(if_metrics['recall_at_1pct_fpr']),\\n",
        "    'recall_at_5pct_fpr': float(if_metrics['recall_at_5pct_fpr']),\\n",
        "    'inference_time_ms': float(if_inference_time)\\n",
        "}\\n",
        "with open(ARTIFACTS / 'isolation_results.json', 'w') as f:\\n",
        "    json.dump(if_results_summary, f, indent=2)\\n",
        "print(\\\"✅ IF artifacts saved\\\")"
    ])
])

# ================================
# LOF
# ================================
new_cells.extend([
    create_markdown_cell([
        "---\\n",
        "# 10. Model 2 — Local Outlier Factor"
    ]),
    create_markdown_cell([
        "## 🎯 LOF Overview\\n",
        "\\n",
        "**How it works:** Measures local density deviation vs neighbors\\n",
        "\\n",
        "**Advantages:**\\n",
        "- ✅ Detects local anomalies\\n",
        "- ✅ Good for varying density clusters\\n",
        "\\n",
        "**Caveats:**\\n",
        "- ⚠️ O(n²) complexity\\n",
        "- ⚠️ **Solution:** Use PCA-reduced features (25D)\\n",
        "\\n",
        "**Hyperparameters:**\\n",
        "- `n_neighbors`: [10, 20, 35, 50]\\n",
        "- `leaf_size`: [30, 50]\\n",
        "\\n",
        "**Critical:** `novelty=True` for scoring new data"
    ]),
    create_code_cell([
        "# LOF hyperparameter grid search\\n",
        "lof_param_grid = {'n_neighbors': [10, 20, 35, 50], 'leaf_size': [30, 50]}\\n",
        "\\n",
        "print(\\\"LOF Grid Search (PCA-reduced data)\\\")\\n",
        "best_lof_score = 0\\n",
        "best_lof_params = None\\n",
        "best_lof_model = None\\n",
        "\\n",
        "param_combinations = list(itertools.product(lof_param_grid['n_neighbors'], lof_param_grid['leaf_size']))\\n",
        "\\n",
        "for n_neigh, leaf in tqdm(param_combinations, desc=\\\"LOF Tuning\\\"):\\n",
        "    model = LocalOutlierFactor(n_neighbors=n_neigh, leaf_size=leaf, novelty=True, n_jobs=-1)\\n",
        "    model.fit(X_train_pca)\\n",
        "    val_scores = -model.score_samples(X_val_pca)\\n",
        "    roc_auc = roc_auc_score(y_val, val_scores)\\n",
        "    \\n",
        "    if roc_auc > best_lof_score:\\n",
        "        best_lof_score = roc_auc\\n",
        "        best_lof_params = {'n_neighbors': n_neigh, 'leaf_size': leaf}\\n",
        "        best_lof_model = model\\n",
        "\\n",
        "print(f\\\"\\\\n✅ Best LOF: {best_lof_params}, ROC-AUC={best_lof_score:.4f}\\\")"
    ]),
    create_code_cell([
        "# Evaluate LOF\\n",
        "test_scores_lof = -best_lof_model.score_samples(X_test_pca_final)\\n",
        "lof_metrics = calculate_metrics(y_test_final, test_scores_lof, \\\"LOF\\\")\\n",
        "print_metrics_summary(lof_metrics)\\n",
        "\\n",
        "lof_inference_time = measure_inference_time(best_lof_model, X_test_pca_final, n_samples=1000, method='score_samples')\\n",
        "print(f\\\"\\\\n⏱️ Inference: {lof_inference_time:.4f} ms/sample\\\")"
    ]),
    create_code_cell([
        "# LOF Visualizations (similar to IF)\\n",
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\\n",
        "axes[0, 0].plot(lof_metrics['fpr'], lof_metrics['tpr'], 'b-', lw=2, label=f\\\"ROC (AUC={lof_metrics['roc_auc']:.4f})\\\")\\n",
        "axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)\\n",
        "axes[0, 0].set_title('LOF - ROC')\\n",
        "axes[0, 0].legend()\\n",
        "axes[0, 0].grid(True, alpha=0.3)\\n",
        "axes[0, 1].plot(lof_metrics['recall'], lof_metrics['precision'], 'g-', lw=2, label=f\\\"PR (AUC={lof_metrics['pr_auc']:.4f})\\\")\\n",
        "axes[0, 1].set_title('LOF - PR')\\n",
        "axes[0, 1].legend()\\n",
        "axes[0, 1].grid(True, alpha=0.3)\\n",
        "normal_scores = test_scores_lof[y_test_final == 0]\\n",
        "attack_scores = test_scores_lof[y_test_final == 1]\\n",
        "axes[1, 0].hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)\\n",
        "axes[1, 0].hist(attack_scores, bins=50, alpha=0.6, label='Attack', color='red', density=True)\\n",
        "axes[1, 0].set_title('LOF - Scores')\\n",
        "axes[1, 0].legend()\\n",
        "cm_1pct, _ = get_confusion_matrix_at_fpr(y_test_final, test_scores_lof, 0.01)\\n",
        "sns.heatmap(cm_1pct, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1], xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])\\n",
        "axes[1, 1].set_title('Confusion @ 1% FPR')\\n",
        "plt.tight_layout()\\n",
        "plt.savefig(FIGURES / 'lof.png', dpi=150, bbox_inches='tight')\\n",
        "plt.show()\\n",
        "print(f\\\"\\\\n💾 Saved: {FIGURES / 'lof.png'}\\\")"
    ]),
    create_code_cell([
        "# Save LOF artifacts\\n",
        "joblib.dump(best_lof_model, ARTIFACTS / 'lof.joblib')\\n",
        "lof_results_summary = {\\n",
        "    'model': 'LOF',\\n",
        "    'best_params': best_lof_params,\\n",
        "    'roc_auc': float(lof_metrics['roc_auc']),\\n",
        "    'pr_auc': float(lof_metrics['pr_auc']),\\n",
        "    'precision_at_100': float(lof_metrics['precision_at_100']),\\n",
        "    'precision_at_1pct': float(lof_metrics['precision_at_1pct']),\\n",
        "    'recall_at_1pct_fpr': float(lof_metrics['recall_at_1pct_fpr']),\\n",
        "    'recall_at_5pct_fpr': float(lof_metrics['recall_at_5pct_fpr']),\\n",
        "    'inference_time_ms': float(lof_inference_time)\\n",
        "}\\n",
        "with open(ARTIFACTS / 'lof_results.json', 'w') as f:\\n",
        "    json.dump(lof_results_summary, f, indent=2)\\n",
        "print(\\\"✅ LOF artifacts saved\\\")"
    ])
])

nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4)

print(f"✅ Added {len(new_cells)} cells (Isolation Forest + LOF)")
print(f"Notebook now has {len(nb['cells'])} cells")
print("💾 Saved updated notebook")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Load data
df = pd.read_csv("bard1_features_with_af.csv")
print("Shape:", df.shape)
print("Class distribution:\n", df['label'].value_counts())

# Replace infinities and large values
# Replace -inf with -10 (since log10 of smallest positive is ~ -6, -10 is safe)
df = df.replace([np.inf, -np.inf], np.nan)
# For log_af, fill NaN with a very negative number (or median)
if 'log_af' in df.columns:
    df['log_af'] = df['log_af'].fillna(-10)
if 'joint_af' in df.columns:
    df['joint_af'] = df['joint_af'].fillna(1e-6)

# Also check for any other columns that might have infinities
for col in df.columns:
    if df[col].dtype in ['float64', 'float32']:
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'class_weight': ['balanced', {0:1, 1:2}]
}

# Grid search with recall scoring
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring='recall',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1,
    error_score='raise'
)

print("Starting grid search...")
grid.fit(X_train, y_train)

print("\nBest parameters:", grid.best_params_)
print("Best cross-validation recall:", grid.best_score_)

# Evaluate best model
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print("\n--- Test set evaluation (threshold=0.5) ---")
print(f"Accuracy: {best_rf.score(X_test, y_test):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Effect of lowering threshold
thresholds = [0.3, 0.4, 0.5]
print("\n--- Effect of lowering threshold ---")
for thresh in thresholds:
    y_pred_adj = (y_proba > thresh).astype(int)
    report = classification_report(y_test, y_pred_adj, output_dict=True)
    recall = report['1']['recall']
    precision = report['1']['precision']
    print(f"Threshold {thresh}: Pathogenic recall = {recall:.3f}, Precision = {precision:.3f}")

# Confusion matrix for threshold 0.4
y_pred_adj = (y_proba > 0.4).astype(int)
cm = confusion_matrix(y_test, y_pred_adj)
print("\nConfusion matrix (threshold=0.4):")
print(cm)
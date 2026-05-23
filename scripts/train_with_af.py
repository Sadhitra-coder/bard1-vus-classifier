import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

# Get the workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

df = pd.read_csv(os.path.join(WORKSPACE_ROOT, "data", "processed", "bard1_features_with_phylop.csv"))
print("Shape:", df.shape)
print("Class distribution:\n", df['label'].value_counts())

X = df.drop(columns=['label'])
y = df['label']

# Replace inf with NaN, then fill NaN appropriately
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    if X[col].isnull().any():
        if col == 'log_af':
            # For log_af, fill with -10 (representing 1e-10 frequency)
            X[col] = X[col].fillna(-10)
        else:
            X[col] = X[col].fillna(X[col].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

print(f"Accuracy: {rf.score(X_test, y_test):.3f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print(f"5-fold CV AUC: {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f})")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Benign', 'Pathogenic'])
disp.plot()
plt.title("Confusion Matrix (with gnomAD AF)")
confusion_path = os.path.join(WORKSPACE_ROOT, "results", "confusion_matrix_with_af.png")
plt.savefig(confusion_path, dpi=150)
print(f"Saved confusion matrix to {confusion_path}")
plt.show()

importances = rf.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.barh(range(len(features)), importances[indices])
plt.yticks(range(len(features)), [features[i] for i in indices])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance (with gnomAD AF)")
plt.gca().invert_yaxis()
plt.tight_layout()
importance_path = os.path.join(WORKSPACE_ROOT, "results", "feature_importance_with_af.png")
plt.savefig(importance_path, dpi=150)
print(f"Saved feature importance to {importance_path}")
plt.show()

model_path = os.path.join(WORKSPACE_ROOT, "models", "bard1_rf_final.pkl")
joblib.dump(rf, model_path)
print(f"Saved model to {model_path}")
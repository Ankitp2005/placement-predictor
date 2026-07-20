"""
train_model.py
==============
Trains a LightGBM classifier on the regenerated placement dataset,
evaluates it honestly on a held-out test set, prints diagnostics,
and saves model.pkl.
"""

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = Path('placement_dataset_realistic.csv')
MODEL_PATH = Path('model.pkl')

FEATURE_COLS = [
    'CGPA', 'Major Projects', 'Workshops/Certificatios', 'Mini Projects',
    'Skills', 'Communication Skill Rating', 'Internship', 'Hackathon',
    '12th Percentage', '10th Percentage', 'backlogs',
]

TARGET_COL = 'PlacementStatus'
ELIGIBILITY_CUTOFF = 6.0   # hard gate — do not train on ineligible students


# ── Load & prepare ──────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Drop ineligible students (CGPA < 6.0) — the backend handles them with
    # an eligibility message, not a model prediction
    df = df[df['CGPA'] >= ELIGIBILITY_CUTOFF].copy()
    print(f"Training on {len(df)} eligible students (CGPA >= {ELIGIBILITY_CUTOFF})")

    # Encode binary categorical columns
    df['Internship'] = (df['Internship'] == 'Yes').astype(int)
    df['Hackathon']  = (df['Hackathon']  == 'Yes').astype(int)

    X = df[FEATURE_COLS]
    y = (df[TARGET_COL] == 'Placed').astype(int)
    return X, y


# ── Train ────────────────────────────────────────────────────────────────────

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    # ── Cross-validation ──
    print("\n=== 5-FOLD CROSS-VALIDATION ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"  ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    # ── Final fit ──
    model.fit(X_train, y_train)

    # ── Held-out test set ──
    print("\n=== HELD-OUT TEST SET EVALUATION ===")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print("\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=['NotPlaced','Placed']))
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    # ── Feature importances ──
    print("\n=== FEATURE IMPORTANCES (gain) ===")
    importances = pd.Series(
        model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = '#' * int(imp / importances.max() * 30)
        print(f"  {feat:<30} {bar} {imp:.0f}")

    # ── Training range report (for app.py TRAINING_RANGES) ──
    X_full = pd.concat([X_train, X_test])
    print("\n=== TRAINING RANGES (copy into app.py TRAINING_RANGES) ===")
    num_cols = [c for c in FEATURE_COLS if c not in ('Internship','Hackathon')]
    for col in num_cols:
        lo = X_full[col].min()
        hi = X_full[col].max()
        print(f'    "{col}": ({lo}, {hi}),')

    return model, X_test, y_test, y_proba


# ── Sanity checks ─────────────────────────────────────────────────────────

def sanity_check(model):
    """Run a few hand-crafted profiles through the model and print results."""
    profiles = [
        dict(label="Star student  (CGPA 9.0, 0 bl, skills 9, intern+hack)",
             CGPA=9.0, backlogs=0, Skills=9, comm=4.7,
             intern=1, hack=1, mini=3, major=2, ws=3, g12=88, g10=86),
        dict(label="Good student  (CGPA 8.0, 0 bl, skills 8, intern, no hack)",
             CGPA=8.0, backlogs=0, Skills=8, comm=4.3,
             intern=1, hack=0, mini=2, major=1, ws=2, g12=78, g10=80),
        dict(label="Average       (CGPA 7.5, 0 bl, skills 6, no intern, no hack)",
             CGPA=7.5, backlogs=0, Skills=6, comm=3.5,
             intern=0, hack=0, mini=1, major=0, ws=1, g12=65, g10=68),
        dict(label="2 backlogs    (CGPA 7.5, 2 bl, skills 8, intern+hack)",
             CGPA=7.5, backlogs=2, Skills=8, comm=4.3,
             intern=1, hack=1, mini=2, major=1, ws=2, g12=72, g10=74),
        dict(label="1 backlog     (CGPA 7.0, 1 bl, skills 7, intern, no hack)",
             CGPA=7.0, backlogs=1, Skills=7, comm=4.0,
             intern=1, hack=0, mini=1, major=0, ws=1, g12=68, g10=70),
        dict(label="Weak + 3 bl   (CGPA 6.5, 3 bl, skills 7, no intern+hack)",
             CGPA=6.5, backlogs=3, Skills=7, comm=3.5,
             intern=0, hack=1, mini=0, major=0, ws=0, g12=60, g10=62),
        dict(label="Min eligible  (CGPA 6.0, 0 bl, skills 6, nothing extra)",
             CGPA=6.0, backlogs=0, Skills=6, comm=3.0,
             intern=0, hack=0, mini=0, major=0, ws=0, g12=55, g10=57),
    ]

    rows = []
    for p in profiles:
        rows.append({
            'CGPA': p['CGPA'], 'Major Projects': p['major'],
            'Workshops/Certificatios': p['ws'], 'Mini Projects': p['mini'],
            'Skills': p['Skills'], 'Communication Skill Rating': p['comm'],
            'Internship': p['intern'], 'Hackathon': p['hack'],
            '12th Percentage': p['g12'], '10th Percentage': p['g10'],
            'backlogs': p['backlogs'],
        })

    X_demo = pd.DataFrame(rows)[FEATURE_COLS]
    probs = model.predict_proba(X_demo)[:, 1]

    print("\n=== SANITY CHECK: HAND-CRAFTED PROFILES ===")
    for p, prob in zip(profiles, probs):
        verdict = 'PLACED' if prob >= 0.5 else 'NOT PLACED'
        print(f"  {prob*100:5.1f}%  {verdict}  — {p['label']}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    X, y = load_data()
    model, X_test, y_test, y_proba = train(X, y)
    sanity_check(model)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")

"""Retrain Parkinson's IF on UCI UPDRS data, matching voice_analyzer feature order:
[jitter_pct, shimmer_db, hnr, rpde, dfa, pitch_mean, pitch_std]
"""
import pathlib, numpy as np, joblib, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

CSV  = pathlib.Path("C:/Users/dell/Desktop/fyp/real_fyp/datasets/speech/uci_parkinsons/parkinsons_updrs.data")
OUT  = pathlib.Path("C:/Users/dell/Desktop/fyp/real_fyp/weights/isolation_forest")

df = pd.read_csv(CSV)
print(f"UCI UPDRS shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Map UCI columns to voice_analyzer feature names
# voice_analyzer keys: jitter_pct, shimmer_db, hnr, rpde, dfa, pitch_mean, pitch_std
# UCI columns:         Jitter(%)   Shimmer(dB) HNR  RPDE  DFA  (no pitch cols)

df = df.dropna(subset=["Jitter(%)", "Shimmer(dB)", "HNR", "RPDE", "DFA"])

# For pitch_mean and pitch_std: typical adult speech is 85-255 Hz (mean ~150 Hz)
# Parkinson's patients show reduced pitch range and slightly lower mean
# Use total_UPDRS to modulate: higher UPDRS → lower pitch mean, lower std
# This is an approximation since actual audio isn't available
threshold = df["total_UPDRS"].median()
label = (df["total_UPDRS"] >= threshold).astype(int).values

# Typical pitch stats (Hz) based on Parkinson's speech literature:
# Controls: mean ~150Hz, std ~25Hz
# Parkinson's: mean ~130Hz, std ~15Hz (reduced range, monotone voice)
pitch_mean = np.where(label == 0,
    np.random.normal(150, 20, len(label)),   # controls
    np.random.normal(130, 20, len(label))    # parkinson's
)
pitch_std = np.where(label == 0,
    np.random.normal(25, 8, len(label)),     # controls
    np.random.normal(15, 6, len(label))      # parkinson's
)
pitch_mean = np.clip(pitch_mean, 80, 300)
pitch_std  = np.clip(pitch_std, 5, 60)

np.random.seed(42)
X = np.column_stack([
    df["Jitter(%)"].values,
    df["Shimmer(dB)"].values,
    df["HNR"].values,
    df["RPDE"].values,
    df["DFA"].values,
    pitch_mean,
    pitch_std,
])
print(f"\nFeature matrix: {X.shape}")
print(f"Labels - normal: {(label==0).sum()}, parkinson's: {(label==1).sum()}")
print(f"Feature means (normal):     {X[label==0].mean(axis=0).round(4)}")
print(f"Feature means (parkinsons): {X[label==1].mean(axis=0).round(4)}")

# Train on normal (lower UPDRS) half only
X_normal = X[label == 0]
print(f"\nTraining IF on {len(X_normal)} normal samples...")
scaler = StandardScaler()
Xs = scaler.fit_transform(X_normal)
model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
model.fit(Xs)

OUT.mkdir(parents=True, exist_ok=True)
joblib.dump(model,  OUT / "parkinsons_prior.pkl")
joblib.dump(scaler, OUT / "parkinsons_scaler.pkl")
print("Saved: parkinsons_prior.pkl + parkinsons_scaler.pkl")

# Sanity check AUC
raw_n = model.decision_function(scaler.transform(X[label==0]))
raw_p = model.decision_function(scaler.transform(X[label==1]))
scores_all = np.concatenate([-raw_n, -raw_p])
labels_all = np.concatenate([np.zeros(len(raw_n)), np.ones(len(raw_p))])
auc = roc_auc_score(labels_all, scores_all)
print(f"AUC (normal=0, parkinson=1): {auc:.4f}")
print(f"Decision scores - normal mean: {raw_n.mean():.4f}, parkinson mean: {raw_p.mean():.4f}")

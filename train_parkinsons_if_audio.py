"""Retrain Parkinson's IF using actual proxy features extracted from real TORGO audio.

Since we don't have a labelled Parkinson's audio dataset with real PD patients,
we train the prior on TORGO controls (normal speech) only — same as dysarthria.
This ensures the scaler/feature distributions match what voice_analyzer.py produces.
The IF will then flag deviations from normal speech as anomalous.
"""
import pathlib, sys, numpy as np, joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "C:/Users/dell/Desktop/fyp/real_fyp/stroke-shield/backend")

TORGO   = pathlib.Path("C:/Users/dell/Desktop/fyp/real_fyp/datasets/speech/torgo")
OUT     = pathlib.Path("C:/Users/dell/Desktop/fyp/real_fyp/weights/isolation_forest")

import soundfile as sf
from app.services.voice_analyzer import VoiceAnalyzer

analyzer = VoiceAnalyzer()
feature_keys = analyzer._parkinsons_feature_keys()
print(f"Parkinson's features ({len(feature_keys)}): {feature_keys}")

rows, labels = [], []
for label_dir, label in [("controls", 0), ("dysarthric", 1)]:
    base = TORGO / label_dir
    wav_files = list(base.rglob("*.wav"))
    print(f"\n{label_dir}: {len(wav_files)} files")
    ok = 0
    for wav_path in wav_files:
        try:
            samples, sr = sf.read(str(wav_path), always_2d=False)
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if len(samples) < sr * 0.5:
                continue
            feats = analyzer.extract_parkinsons_features(samples, sr)
            vec = np.array([feats[k] for k in feature_keys], dtype=float)
            if not np.all(np.isfinite(vec)):
                continue
            rows.append(vec)
            labels.append(label)
            ok += 1
        except Exception:
            continue
    print(f"  processed: {ok}")

X = np.vstack(rows)
y = np.array(labels)
print(f"\nTotal: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Controls: {(y==0).sum()}, Dysarthric: {(y==1).sum()}")
print(f"Feature means (controls): {X[y==0].mean(axis=0).round(4)}")
print(f"Feature means (dysarth):  {X[y==1].mean(axis=0).round(4)}")

# Train on controls only
X_controls = X[y == 0]
print(f"\nTraining IF on {len(X_controls)} control samples...")
scaler = StandardScaler()
Xs = scaler.fit_transform(X_controls)
model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
model.fit(Xs)

OUT.mkdir(parents=True, exist_ok=True)
joblib.dump(model,  OUT / "parkinsons_prior.pkl")
joblib.dump(scaler, OUT / "parkinsons_scaler.pkl")
print("Saved: parkinsons_prior.pkl + parkinsons_scaler.pkl")

raw_ctrl = model.decision_function(scaler.transform(X[y==0]))
raw_dys  = model.decision_function(scaler.transform(X[y==1]))
print(f"\nDecision scores:")
print(f"  Controls mean:    {raw_ctrl.mean():.4f}")
print(f"  Dysarthric mean:  {raw_dys.mean():.4f}")
scores_all = np.concatenate([-raw_ctrl, -raw_dys])
labels_all = np.concatenate([np.zeros(len(raw_ctrl)), np.ones(len(raw_dys))])
auc = roc_auc_score(labels_all, scores_all)
print(f"  AUC: {auc:.4f}")

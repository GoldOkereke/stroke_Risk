import pathlib, numpy as np, joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

AFLFP = pathlib.Path("C:/Users/dell/Desktop/fyp/real_fyp/datasets/facial/facial_palsy_db/AFLFP")
OUT   = pathlib.Path("C:/Users/dell/Desktop/fyp/real_fyp/weights/isolation_forest")

def parse_pts(path):
    pts = []
    for line in path.read_text(errors="ignore").splitlines():
        p = line.strip().split()
        if len(p) >= 2:
            try:
                pts.append([float(p[0]), float(p[1])])
            except ValueError:
                pass
    return np.array(pts, dtype=float)

def extract5(pts):
    if len(pts) < 68:
        return None
    fh = abs(pts[8, 1] - pts[27, 1])
    if fh < 1e-6:
        fh = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    if fh < 1e-6:
        return None
    mouth_droop = abs(pts[48, 1] - pts[54, 1]) / fh
    eye_droop   = abs(pts[36, 1] - pts[45, 1]) / fh
    brow_asym   = abs(pts[22:27, 1].mean() - pts[17:22, 1].mean()) / fh
    nose        = pts[30]
    nasol       = abs(np.linalg.norm(nose - pts[48]) - np.linalg.norm(nose - pts[54])) / fh
    ul_y        = pts[51, 1]
    smile_sym   = abs((pts[48, 1] - ul_y) - (pts[54, 1] - ul_y)) / fh
    return np.clip([mouth_droop, eye_droop, brow_asym, nasol, smile_sym], 0.0, 1.0)

rows = []
pts_files = list(AFLFP.rglob("*.pts"))
print(f"Processing {len(pts_files)} .pts files...")
for pts_path in pts_files:
    pts = parse_pts(pts_path)
    f = extract5(pts)
    if f is not None:
        rows.append(f)

X = np.vstack(rows)
print(f"AFLFP samples: {X.shape[0]}, features: {X.shape[1]}")
print(f"Feature means: {X.mean(axis=0).round(4)}")
print(f"Feature stds:  {X.std(axis=0).round(4)}")
print(f"Feature mins:  {X.min(axis=0).round(4)}")
print(f"Feature maxs:  {X.max(axis=0).round(4)}")

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
model.fit(Xs)

OUT.mkdir(parents=True, exist_ok=True)
joblib.dump(model,  OUT / "facial_prior.pkl")
joblib.dump(scaler, OUT / "facial_scaler.pkl")
print("Saved: facial_prior.pkl + facial_scaler.pkl")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
import pickle
import json

# Cleveland Heart Disease Dataset features (UCI)
# We'll generate synthetic data based on real distributions
np.random.seed(42)
n = 1000

def generate_heart_data(n):
    data = []
    for _ in range(n):
        target = np.random.choice([0, 1], p=[0.54, 0.46])
        
        age = np.random.normal(60 if target else 52, 9)
        age = max(29, min(77, int(age)))
        
        sex = np.random.choice([0, 1], p=[0.3, 0.7] if target else [0.35, 0.65])
        
        cp_probs = [0.07, 0.16, 0.28, 0.49] if target else [0.47, 0.25, 0.17, 0.11]
        cp = np.random.choice([0, 1, 2, 3], p=cp_probs)
        
        trestbps = np.random.normal(134 if target else 129, 18)
        trestbps = max(94, min(200, int(trestbps)))
        
        chol = np.random.normal(251 if target else 242, 50)
        chol = max(126, min(564, int(chol)))
        
        fbs = np.random.choice([0, 1], p=[0.82, 0.18] if target else [0.86, 0.14])
        
        restecg = np.random.choice([0, 1, 2], p=[0.39, 0.55, 0.06] if target else [0.58, 0.38, 0.04])
        
        thalach = np.random.normal(139 if target else 158, 22)
        thalach = max(71, min(202, int(thalach)))
        
        exang = np.random.choice([0, 1], p=[0.42, 0.58] if target else [0.77, 0.23])
        
        oldpeak = np.random.exponential(1.8 if target else 0.7)
        oldpeak = round(max(0, min(6.2, oldpeak)), 1)
        
        slope = np.random.choice([0, 1, 2], p=[0.32, 0.38, 0.30] if target else [0.07, 0.45, 0.48])
        
        ca = np.random.choice([0, 1, 2, 3], p=[0.27, 0.35, 0.25, 0.13] if target else [0.72, 0.16, 0.08, 0.04])
        
        thal_probs = [0.05, 0.06, 0.70, 0.19] if target else [0.05, 0.12, 0.36, 0.47]
        thal = np.random.choice([0, 1, 2, 3], p=thal_probs)
        
        data.append([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target])
    
    cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
    return pd.DataFrame(data, columns=cols)

df = generate_heart_data(n)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train ensemble
rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

rf.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)
lr.fit(X_train_scaled, y_train)

# Evaluate
models = {'RandomForest': rf, 'GradientBoosting': gb, 'LogisticRegression': lr}
results = {}
for name, model in models.items():
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    acc = (preds == y_test).mean()
    auc = roc_auc_score(y_test, probs)
    cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    results[name] = {'accuracy': round(acc*100, 1), 'auc': round(auc, 3), 'cv_accuracy': round(cv*100, 1)}
    print(f"{name}: Acc={acc:.3f}, AUC={auc:.3f}, CV={cv:.3f}")

# Feature importances from RF
feature_names = X.columns.tolist()
importances = rf.feature_importances_
feat_imp = dict(zip(feature_names, [round(float(v)*100, 1) for v in importances]))

# Save
with open('heart_model.pkl', 'wb') as f:
    pickle.dump({'rf': rf, 'gb': gb, 'lr': lr, 'scaler': scaler}, f)

with open('model_info.json', 'w') as f:
    json.dump({'results': results, 'feature_importances': feat_imp}, f)

print("\nModel saved!")
print(json.dumps(results, indent=2))
print("\nFeature importances:")
for k, v in sorted(feat_imp.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}%")

from flask import Flask, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

with open('heart_model.pkl', 'rb') as f:
    models = pickle.load(f)

with open('model_info.json', 'r') as f:
    model_info = json.load(f)

rf = models['rf']
gb = models['gb']
lr = models['lr']
scaler = models['scaler']

FEATURE_NAMES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

RISK_FACTORS = {
    'cp': 'Chest Pain Type',
    'age': 'Age',
    'ca': 'Major Vessels (Fluoroscopy)',
    'thalach': 'Max Heart Rate',
    'oldpeak': 'ST Depression',
    'exang': 'Exercise-Induced Angina',
    'chol': 'Cholesterol',
    'trestbps': 'Resting Blood Pressure',
    'thal': 'Thalassemia',
    'slope': 'ST Slope',
}

def get_risk_level(prob):
    if prob < 0.25:
        return 'Low', '#1D9E75'
    elif prob < 0.5:
        return 'Moderate', '#EF9F27'
    elif prob < 0.75:
        return 'High', '#D85A30'
    else:
        return 'Very High', '#E24B4A'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [float(data.get(f, 0)) for f in FEATURE_NAMES]
    X = np.array([features])
    X_scaled = scaler.transform(X)
    
    rf_prob = rf.predict_proba(X_scaled)[0][1]
    gb_prob = gb.predict_proba(X_scaled)[0][1]
    lr_prob = lr.predict_proba(X_scaled)[0][1]
    ensemble_prob = (rf_prob + gb_prob + lr_prob) / 3
    
    risk_level, risk_color = get_risk_level(ensemble_prob)
    
    # Top contributing risk factors
    feat_imp = rf.feature_importances_
    feat_values = {FEATURE_NAMES[i]: float(feat_imp[i]) for i in range(len(FEATURE_NAMES))}
    
    contributing = []
    for feat, imp in sorted(feat_values.items(), key=lambda x: -x[1])[:5]:
        val = data.get(feat, 0)
        contributing.append({'feature': feat, 'name': RISK_FACTORS.get(feat, feat), 'importance': round(imp*100, 1), 'value': val})
    
    return jsonify({
        'probability': round(ensemble_prob * 100, 1),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'prediction': int(ensemble_prob > 0.5),
        'model_breakdown': {
            'random_forest': round(rf_prob * 100, 1),
            'gradient_boosting': round(gb_prob * 100, 1),
            'logistic_regression': round(lr_prob * 100, 1)
        },
        'contributing_factors': contributing
    })

@app.route('/model-info', methods=['GET'])
def get_model_info():
    return jsonify(model_info)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5050, debug=False)

import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- 1. LOAD ASSETS ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "Employee_Attrition_Risk_Analysis_&_Prediction.pkl")
ohe_path = os.path.join(base_path, "OneHotEncoder.pkl")

try:
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    with open(ohe_path, "rb") as f:
        OneHot_model = pickle.load(f)
    print("✅ Files loaded successfully.")
except Exception as e:
    print(f"❌ Error loading pickle files: {e}")

# --- 2. SERVE THE UI ---
@app.route('/')
def home():
    return render_template('index.html')

# --- 3. PREDICTION LOGIC ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_input = request.get_json()
        
        # Create DataFrame from raw JSON input
        user_df = pd.DataFrame([raw_input])

        # --- FIX DATA TYPES (Convert 'object' to 'int') ---
        # This prevents the error you saw in your screenshots
        cols_to_fix = [
            'Age', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
            'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
            'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        
        for col in cols_to_fix:
            if col in user_df.columns:
                user_df[col] = pd.to_numeric(user_df[col], errors='coerce').fillna(0).astype(int)

        # Mirroring your Jupyter encoding
        user_df['Gender'] = user_df['Gender'].map({'Male': 1, 'Female': 0})
        user_df['OverTime'] = user_df['OverTime'].map({'Yes': 1, 'No': 0})
        user_df['BusinessTravel'] = user_df['BusinessTravel'].map({
             'Travel_Rarely': 1, 'Travel_Frequently': 0, 'Non-Travel': 2
        })

        cat_cols = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
        
        # Transform categorical columns using your loaded OneHotEncoder
        ohe_features = OneHot_model.transform(user_df[cat_cols])
        if hasattr(ohe_features, "toarray"):
            ohe_features = ohe_features.toarray()
            
        ohe_df = pd.DataFrame(ohe_features, columns=OneHot_model.get_feature_names_out(cat_cols))

        # Drop original text columns
        user_numeric = user_df.drop(columns=cat_cols)
        user_final = pd.concat([user_numeric, ohe_df], axis=1)

        # Align columns with the EXACT order from training (Your 27 columns)
        user_final = user_final.reindex(columns=loaded_model.feature_names_in_, fill_value=0)

        # Run Prediction
        probability = loaded_model.predict_proba(user_final)[0][1]
        risk_score = round(float(probability) * 100, 2)

        return jsonify({
            "status": "Success",
            "risk_percentage": risk_score,
            "prediction": "High Risk" if risk_score > 50 else "Low Risk"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "Failed"}), 500

if __name__ == '__main__':
    # Port 7860 is required for Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860, debug=True)

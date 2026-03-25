import pandas as pd
import numpy as np
import pickle
import os
import shap
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- 1. CONFIGURE GEMINI AI ---
# It's best practice to use environment variables for keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY") 
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 2. LOAD ASSETS ---
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "Employee_Attrition_Risk_Analysis_&_Prediction.pkl")
ohe_path = os.path.join(base_path, "OneHotEncoder.pkl")

try:
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    with open(ohe_path, "rb") as f:
        OneHot_model = pickle.load(f)
    
    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(loaded_model)
    print("✅ Files and SHAP Explainer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")

# --- 3. GEMINI REAL-TIME SUGGESTION ENGINE ---
def get_gemini_suggestions(risk_score, top_factors_list):
    """
    Sends the SHAP factors to Gemini to get tailored HR solutions.
    """
    factors_str = ", ".join([f"{f['feature']}" for f in top_factors_list])
    
    prompt = f"""
    As an expert HR Consultant, I have an employee with a {risk_score}% attrition risk.
    The top 5 predictive factors for this risk are: {factors_str}.
    
    Provide 5 brief, actionable, and professional strategies to retain this employee.
    Keep each strategy under 15 words and make them specific to the factors mentioned.
    Format: Return only the 5 bullet points.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        # Split text into a list and clean up bullets/empty lines
        suggestions = [line.strip().replace('* ', '') for line in response.text.strip().split('\n') if line.strip()]
        return suggestions[:5]
    except Exception as e:
        print(f"Gemini Error: {e}")
        return ["Focus on employee engagement", "Review compensation", "Improve work-life balance", "Career growth planning", "Enhance workplace culture"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_input = request.get_json()
        user_df = pd.DataFrame([raw_input])

        # --- 1. TEXT MAPPINGS ---
        education_map = {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
        job_level_map = {'Entry Level': 1, 'Intermediate / Junior': 2, 'Mid-Level / Specialist': 3, 'Senior Management': 4, 'Executive / VP': 5}

        if 'Education' in user_df.columns and user_df['Education'].iloc[0] in education_map:
            user_df['Education'] = user_df['Education'].map(education_map)
        
        if 'JobLevel' in user_df.columns and user_df['JobLevel'].iloc[0] in job_level_map:
            user_df['JobLevel'] = user_df['JobLevel'].map(job_level_map)

        # --- 2. DATA TYPE CONVERSION ---
        cols_to_fix = [
            'Age', 'DistanceFromHome', 'MonthlyIncome', 'TotalWorkingYears', 
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
            'YearsWithCurrManager', 'PercentSalaryHike', 'NumCompaniesWorked', 
            'TrainingTimesLastYear', 'JobSatisfaction', 'EnvironmentSatisfaction', 
            'JobInvolvement', 'RelationshipSatisfaction', 'WorkLifeBalance', 
            'PerformanceRating', 'StockOptionLevel'
        ]
        
        for col in cols_to_fix:
            if col in user_df.columns:
                user_df[col] = pd.to_numeric(user_df[col], errors='coerce').fillna(0).astype(int)

        # --- 3. ENCODING ---
        user_df['Gender'] = user_df['Gender'].map({'Male': 1, 'Female': 0})
        user_df['OverTime'] = user_df['OverTime'].map({'Yes': 1, 'No': 0})
        user_df['BusinessTravel'] = user_df['BusinessTravel'].map({'Travel_Rarely': 1, 'Travel_Frequently': 0, 'Non-Travel': 2})

        cat_cols = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
        ohe_features = OneHot_model.transform(user_df[cat_cols])
        if hasattr(ohe_features, "toarray"): ohe_features = ohe_features.toarray()
        ohe_df = pd.DataFrame(ohe_features, columns=OneHot_model.get_feature_names_out(cat_cols))

        user_numeric = user_df.drop(columns=cat_cols)
        user_final = pd.concat([user_numeric, ohe_df], axis=1)
        user_final = user_final.reindex(columns=loaded_model.feature_names_in_, fill_value=0)

        # --- 4. PREDICTION & RISK LOGIC ---
        probability = loaded_model.predict_proba(user_final)[0][1]
        risk_score = round(float(probability) * 100, 2)

        if risk_score < 40:
            category = "Low Risk"
        elif 40 <= risk_score <= 80:
            category = "Moderate Risk"
        else:
            category = "High Risk"

        # --- 5. SHAP EXPLANATION (Top 5 Factors) ---
        shap_values = explainer.shap_values(user_final)
        
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]

        feature_importance = pd.Series(vals, index=user_final.columns)
        top_factors_series = feature_importance.sort_values(ascending=False).head(5)

        # Prepare factors for Gemini
        top_factors_list = []
        for feature, val in top_factors_series.items():
            top_factors_list.append({
                "feature": feature,
                "impact_score": round(float(val), 4)
            })

        # --- 6. GET AI RECOMMENDATIONS ---
        ai_suggestions = get_gemini_suggestions(risk_score, top_factors_list)

        # Combine factors with their respective AI suggestions
        # We zip them so the frontend gets a pair of Factor + Solution
        final_insights = []
        for i in range(len(top_factors_list)):
            final_insights.append({
                "feature": top_factors_list[i]["feature"],
                "impact_score": top_factors_list[i]["impact_score"],
                "suggestion": ai_suggestions[i] if i < len(ai_suggestions) else "Consult HR for tailored strategy"
            })

        return jsonify({
            "status": "Success", 
            "risk_percentage": risk_score, 
            "prediction": category,
            "top_factors": final_insights
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e), "status": "Failed"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)

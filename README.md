# 🚀 AI Employee Attrition Predictor

[![Render Deployment](https://img.shields.io/badge/Render-Live_Demo-00d1b2?style=for-the-badge&logo=render&logoColor=white)](https://employee-xcjt.onrender.com/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/PayalGaikwad/Employee_Risk_Analysis)

An interactive Machine Learning web application designed to identify high-risk employees using **27 key behavioral and professional markers.**

---

## 🌐 Live Deployments
* **Primary (Render):** [https://employee-xcjt.onrender.com/](https://employee-xcjt.onrender.com/)
* **Mirror (Hugging Face):** [Employee Attrition Risk Analysis](https://huggingface.co/spaces/PayalGaikwad/Employee_Risk_Analysis)

---

## 📊 Dataset Insights
The engine is trained on a dataset of **1,470 records** with an **Attrition Rate of 16.1%**.

### **Critical Markers Analyzed:**
* **Work-Life Balance**: OverTime and DistanceFromHome.
* **Economic Factors**: MonthlyIncome and PercentSalaryHike.
* **Loyalty Metrics**: YearsAtCompany and YearsWithCurrManager.

---

## ⚙️ Technical Implementation
The system ensures high accuracy by strictly managing data types and feature alignment:

1. **Input Handling**: The frontend captures 27 features. 
2. **Type Casting**: The backend automatically converts string-based numeric inputs (often received as `object` types) into `int64` to prevent model runtime errors.
3. **Encoding Pipeline**: 
    * Binary Mapping for `Gender` and `OverTime`.
    * One-Hot Encoding for categorical fields like `JobRole` and `Department`.



---

## 🏆 Model Performance
We prioritize **Recall** to ensure that employees likely to leave are not missed by the system.

| Model | Precision (Attrition) | Recall (Attrition) | F1-Score |
| :--- | :---: | :---: | :---: |
| **XGBoost** | **0.50** | **0.41** | **0.46** |
| **Logistic Regression** | 0.50 | 0.36 | 0.42 |
| **Random Forest** | 1.00 | 0.13 | 0.23 |

---

## 🛠️ Tech Stack
* **Backend**: Python (Flask, Gunicorn)
* **Frontend**: HTML5, Tailwind CSS, Chart.js
* **ML Library**: Scikit-learn, XGBoost, Pandas
* **Hosting**: Render & Hugging Face Spaces

---

## 💡 Recommendations
* **Threshold Tuning**: Managers can adjust the risk threshold to be more sensitive to high-value talent.
* **Intervention**: High-risk scores (e.g., >70%) should trigger immediate engagement interviews or salary reviews.

---
**Developed by Payal Gaikwad**

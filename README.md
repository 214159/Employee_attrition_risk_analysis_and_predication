# 🚀 Employee_attrition_risk_analysis_and_predication

[![Render Deployment](https://img.shields.io/badge/Render-Live_Demo-00d1b2?style=for-the-badge&logo=render&logoColor=white)](https://employee-xcjt.onrender.com/https://employee-attrition-risk-analysis-and-lyd4.onrender.com)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue?style=for-the-badge)](https://huggingface.co/spaces/PayalGaikwad/Employee_Risk_Analysis)

An interactive Machine Learning web application designed to identify high-risk employees using **27 key behavioral and professional markers.**

---

## 🖥️ System Interface
<img width="1919" height="907" alt="image" src="https://github.com/user-attachments/assets/d8928243-d7ae-460f-b1df-02b4e6a9d93c" />

*The high-tech interface features a glassmorphism design with real-time risk calculation.*

---
## 🎯 Problem Statement
Employee attrition is a major challenge for modern organizations. High turnover rates lead to:
* 💸 **Increased Recruitment Costs**: Replacing a mid-level employee can cost up to 150% of their annual salary.
* 📉 **Productivity Loss**: New hires take months to reach peak efficiency.
* 🧠 **Knowledge Drain**: Loss of experienced talent and institutional memory.

**The Objective:** To build a predictive engine that identifies high-risk employees *before* they resign, allowing HR to take preventive actions such as salary reviews, role changes, or stay-interviews.

---

## 📊 Dataset & Insights
The model is trained on the **IBM HR Analytics Attrition Dataset**, consisting of **1,470 records** across **35 features** (refined to 27 for production).

### **Key Data Insights (EDA):**
* **The Overtime Trigger**: Employees working Overtime are **3x more likely** to leave.
* **The Salary Gap**: Most attrited employees fall within the lower monthly income brackets (<$5,000).
* **The "Magic" Tenure Mark**: Attrition is highest among employees who have been with the company for **less than 2 years** or have had the **same manager** for a long period without promotion.
* **Age Factor**: Younger employees (Age 18-30) show a significantly higher tendency to switch companies compared to established professionals.

---

## 🌐 Live Deployments
* **Primary (Render):** [https://employee-xcjt.onrender.com/](https://employee-xcjt.onrender.com/)
* **Mirror (Hugging Face):** [Employee Attrition Risk Analysis](https://huggingface.co/spaces/PayalGaikwad/Employee_Risk_Analysis)

---

## 📊 Diagnostic Engine Logic
The engine analyses employee stability through three distinct lenses, using a **Triple-Tier Risk Assessment** model:

### **1. Risk Categorization**
* 🟢 **Low Risk (<40%)**: Stable employee profile.
* 🟠 **Moderate Risk (40% - 80%)**: Warning phase; requires engagement.
* 🔴 **High Risk (>80%)**: Critical flight risk; immediate intervention needed.

<img width="1918" height="915" alt="image" src="https://github.com/user-attachments/assets/01d7df33-3b64-4f5f-b3ff-8e9720dde391" />

### **2. Feature Mapping**
To ensure 100% accuracy, the system uses a custom **Text-to-Numeric Pipeline** for:
* **Education & Job Levels**: Mapped from professional titles to 1-5 scales.
* **Satisfaction Ratings**: Converted from text dropdowns to 1-4 ordinals.

---

## 🏆 Why XGBoost?
After rigorous testing across multiple algorithms, **XGBoost** was selected as the production model.

**The Decision Factor:**
While Random Forest showed high precision, it lacked the ability to catch all potential "leavers" (Low Recall). **XGBoost provided the best balance of Precision and Recall**, ensuring the model doesn't just predict accurately, but actually captures the highest number of high-risk employees before they resign.

| Model | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **XGBoost (Selected)** | **0.50** | **0.41** | **0.46** |
| **Logistic Regression** | 0.50 | 0.36 | 0.42 |
| **Random Forest** | 1.00 | 0.13 | 0.23 |

---

## ⚙️ Technical Implementation
1. **Intelligent UI**: Built with **Red-Box Validation**—empty fields turn red immediately to prevent partial data submission.
2. **Robust Backend**: Data types are force-cast to `int64` to prevent model runtime errors.
3. **Encoding**: Integrated One-Hot Encoding for categorical stability.
---
* **Frontend UI/UX & System Integration**: Developed by **Gemini (AI)**
* **Tech Stack**: Python, Flask, XGBoost, Tailwind CSS, Chart.js
---

## 🏁 Conclusion
The **AI Employee Attrition Predictor** successfully bridges the gap between raw HR data and proactive talent management. By utilizing the XGBoost model's superior recall capabilities, organizations can move from reactive hiring to proactive retention, saving significant costs associated with employee turnover.

---
## 🔮 Future Roadmap
To further enhance the **Neural Attrition Decoder**, the following features are planned:

* **Prescriptive Analytics**: Moving beyond "who will leave" to "what specific action will make them stay."
* **Stay-Interview Automation**: Triggering automated calendar invites for HR when an employee hits the 'Moderate Risk' threshold.
* **Sentiment Analysis**: Integrating NLP to analyze employee feedback from internal surveys.
* **Cloud Scalability**: Migrating to AWS/Azure for real-time processing of large-scale enterprise data.
* **Longitudinal Tracking**: Analyzing how an employee's risk score changes month-over-month to identify seasonal attrition trends.

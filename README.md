# **CDC Diabetes Prediction Project**

## **Overview**

This project leverages machine learning techniques to predict **diabetes risk** using health and lifestyle indicators from the **CDC dataset**. It includes **data preprocessing, exploratory analysis, feature engineering, handling class imbalance, hyperparameter tuning, and model evaluation**.

## **Table of Contents**

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Machine Learning Models](#machine-learning-models)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Results & Insights](#results--insights)
- [Future Improvements](#future-improvements)
- [Installation & Usage](#installation--usage)
- [Contributors](#contributors)
- [License](#license)

---

## **Dataset**

The dataset is sourced from the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** and includes various **health, lifestyle, and demographic attributes**:

- **Target Variable:** `Diabetes_binary` (0 = No Diabetes, 1 = Diabetes).
- **Health Indicators:** `HighBP`, `HighChol`, `CholCheck`, `BMI`, `Stroke`, `HeartDiseaseorAttack`.
- **Lifestyle Factors:** `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`.
- **Healthcare Access:** `AnyHealthcare`, `NoDocbcCost`.
- **Self-Reported Health:** `GenHlth`, `MentHlth`, `PhysHlth`.
- **Demographics:** `Sex`, `Age`, `Education`, `Income`.

---

## **Project Workflow**

✔ **Exploratory Data Analysis (EDA)**: Understand feature distributions, relationships, and correlations.  
✔ **Data Cleaning & Preprocessing**: Handling missing values, encoding categorical variables, and normalizing features.  
✔ **Feature Engineering**: Creating meaningful transformations to improve predictive performance.  
✔ **Handling Imbalanced Data**: Using class weights and resampling techniques.  
✔ **Model Selection & Training**: Comparing multiple machine learning models.  
✔ **Hyperparameter Tuning**: Optimizing models for better performance.  
✔ **Evaluation & Calibration**: Assessing model accuracy, recall, precision, AUC-ROC, and calibration plots.

---

## **Exploratory Data Analysis**

EDA was performed using **Matplotlib, Seaborn, and Pandas Profiling** to visualize key insights:

- **Distribution Analysis:** Histograms and box plots for `BMI`, `Age`, and `GenHlth`.
- **Correlation Matrix:** Identified strong predictors like **Age, BMI, HighBP, and GenHlth**.
- **Target Distribution:** Significant class imbalance (majority non-diabetic).
- **Category Counts:** Bar charts showing prevalence of high blood pressure, cholesterol levels, and physical activity among diabetic individuals.

---

## **Data Preprocessing & Feature Engineering**

- **Categorical Encoding**: Converted categorical variables using **one-hot encoding** or **ordinal encoding**.
- **Feature Scaling**: Normalized numeric features for models sensitive to scale.
- **Outlier Handling**: Detected and treated outliers in `BMI`, `MentHlth`, and `PhysHlth`.
- **Feature Selection**: Used **SHAP values** and feature importance plots to refine inputs.

---

## **Handling Imbalanced Data**

The dataset had **significantly more non-diabetic cases than diabetic cases**. We applied:  
✅ **Class Weights**: Adjusted weights in models like **Logistic Regression** and **XGBoost**.  
✅ **Oversampling & Undersampling**: Experimented with **SMOTE** and **random undersampling**.  
✅ **Threshold Tuning**: Adjusted probability thresholds to **reduce false negatives**.

---

## **Machine Learning Models**

We evaluated multiple models, prioritizing recall due to the **high cost of false negatives** in healthcare predictions.

| Model                    | Accuracy  | Precision | Recall    | F1-score  | ROC-AUC  |
| ------------------------ | --------- | --------- | --------- | --------- | -------- |
| Logistic Regression      | 74.2%     | 41.8%     | 56.3%     | 47.8%     | 0.79     |
| Decision Tree            | 72.5%     | 38.9%     | 59.1%     | 46.9%     | 0.77     |
| Random Forest            | 78.8%     | 52.6%     | 63.4%     | 57.5%     | 0.84     |
| **XGBoost (Best Model)** | **81.1%** | **58.2%** | **85.0%** | **68.9%** | **0.87** |

---

## **Hyperparameter Tuning**

Using **RandomizedSearchCV**, we optimized:

- `n_estimators`, `max_depth`, `learning_rate` for model complexity.
- `subsample`, `colsample_bytree` for regularization.
- `scale_pos_weight` to balance the classes.

---

## **Model Evaluation**

### **Confusion Matrix**

- ✅ **Recall:** 85% of diabetic cases were correctly identified.
- ❌ **False Positives:** Higher rate, meaning some non-diabetic cases were incorrectly flagged.

### **ROC & Precision-Recall Curves**

- **ROC-AUC:** **0.87** → Good discrimination capability.
- **Precision-Recall Curve:** Precision ~58%, indicating a trade-off.

### **Calibration Plot**

The **XGBoost model was under-confident**, suggesting the need for **probability calibration (Platt scaling or Isotonic Regression).**

---

## **Results & Insights**

- **Top Predictors**: `GenHlth`, `HighBP`, `Age`, and `BMI`.
- **Lifestyle features** (`Fruits`, `Veggies`, `PhysActivity`) had **less impact** than expected.
- **Potential for Threshold Tuning**: Adjusting the probability cutoff can **reduce false positives**.
- **Calibrated probabilities** could **enhance real-world interpretability** for healthcare providers.

---

## **Future Improvements**

🚀 **Model Calibration**: Adjust probability estimates to better align with reality.  
🚀 **Ensemble Methods**: Combine multiple models for improved robustness.  
🚀 **More Features**: Include **blood test results or genetic markers** if available.  
🚀 **Deployment**: Convert the model into an **API** for real-time diabetes risk assessment.

---

## **Installation & Usage**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Notebook**

Open **Jupyter Notebook** and execute `diabetes_prediction.ipynb`.

---

## **Contributors**

👤 **Your Name** – [LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## **License**

This project is **MIT Licensed** – feel free to use and modify it!

---

This **README.md** is structured to be clear, professional, and user-friendly. Let me know if you want to tweak anything! 🚀

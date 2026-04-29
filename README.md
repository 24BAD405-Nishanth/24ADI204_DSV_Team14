<div align="center">

# ❤️ Heartbeats & Habits

## Predicting 10-Year Coronary Heart Disease (CHD) Risk

### 24ADI204 – Data Science and Visualization Project

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge\&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?style=for-the-badge\&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge\&logo=scikit-learn)
![PowerBI](https://img.shields.io/badge/PowerBI-Dashboard-yellow?style=for-the-badge\&logo=powerbi)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen?style=for-the-badge)

Transforming raw clinical data into actionable cardiovascular risk intelligence.

</div>

---

# 📌 Overview

**Heartbeats & Habits** is a complete end-to-end Data Science project focused on predicting an individual's **10-year risk of Coronary Heart Disease (CHD)** using the **Framingham Heart Study Dataset**.

This project includes:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Feature Selection
* PCA
* Dashboard Visualization
* Healthcare Insight Generation

---

# 🎯 Objective

To identify high-risk patients using clinical and lifestyle indicators and support preventive healthcare decisions through data-driven analysis.

---

# 📂 Dataset Information

| Attribute       | Value                  |
| --------------- | ---------------------- |
| Dataset Name    | Framingham Heart Study |
| Source          | Kaggle                 |
| Records         | 4,240                  |
| Features        | 16                     |
| Target Variable | `TenYearCHD`           |

### Key Features

`age`, `male`, `education`, `currentSmoker`, `cigsPerDay`, `sysBP`, `diaBP`, `BMI`, `glucose`, `diabetes`, `totChol`

---

# 🛠️ Tech Stack

### Languages & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

### Visualization

* Power BI

### Tools

* Jupyter Notebook
* VS Code

---

# ⚙️ Workflow

## 1️⃣ Data Preprocessing

✔ Missing value imputation
✔ Duplicate checking
✔ Data cleaning
✔ Structured dataset preparation

## 2️⃣ Outlier Handling

Used **IQR Capping** for:

* glucose
* BMI
* sysBP
* diaBP

## 3️⃣ Exploratory Data Analysis

* Histograms
* KDE plots
* Boxplots
* Correlation heatmaps
* Risk distribution analysis

## 4️⃣ Feature Engineering

* `log1p()` transformation
* RobustScaler normalization
* Ordinal encoding

## 5️⃣ Feature Selection

Compared:

* Correlation Analysis
* Mutual Information
* Random Forest Importance

## 6️⃣ Principal Component Analysis

Reduced dimensionality while preserving major variance patterns.

---

# 📈 Major Findings

| Insight             | Observation                     |
| ------------------- | ------------------------------- |
| Strongest Predictor | Systolic Blood Pressure         |
| High Risk Age Group | 60+                             |
| Diabetes Effect     | Nearly 3× higher CHD risk       |
| Hypertension        | Strong positive correlation     |
| Smoking             | Heavy smokers carry higher risk |
| Glucose             | Strong threshold-based impact   |

---

# 📊 Dashboard Highlights

### Power BI Dashboard Includes:

✅ KPI Cards
✅ Risk Segmentation
✅ Age vs CHD Risk
✅ Smoking Analysis
✅ Diabetes Comparison
✅ Blood Pressure Impact
✅ AI Key Influencers

---

# 📁 Project Structure

```bash
24ADI204_DSV_Team14/
│── FinalDocumentation/
│── FinalPresentation/
│── notebook/
│────── data/  
│── powerBI-Dashboard/
│── WeeklyReport/
│── README.md
```

---

# 🚀 Installation

```bash
git clone https://github.com/24BAD405-Nishanth/24ADI204_DSV_Team14.git
cd 24ADI204_DSV_Team14
pip install -r requirements.txt
jupyter notebook
```

---

# 🔗 Repository Links

### Main Repository

https://github.com/Gokulnaath-gif/24ADI204_DSV_Team14

### Mirror Repository

https://github.com/24BAD405-Nishanth/24ADI204_DSV_Team14

---

# 🔮 Future Improvements

* Logistic Regression
* Random Forest
* XGBoost
* SMOTE Balancing
* SHAP Explainability
* Streamlit Deployment

---

# 👨‍💻 Team Members

* Gokulnaath M
* Kamalesh N
* Nishanth P
* Jayaraksha Reguraj

---

# 🏫 Academic Info

**Course:** 24ADI204 – Data Science and Visualization
**Department:** Artificial Intelligence and Data Science
**Institution:** Kumaraguru College of Technology

---

<div align="center">

## ❤️ Turning Healthcare Data into Better Decisions

</div>

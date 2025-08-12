# Lab 4 - Binary Classification Research

**Student:** Gayathri Chinka  
**Dataset:** Company Bankruptcy Prediction  
**Task:** Research-based decision making for bankruptcy prediction models

---

## 📁 Project Structure

```
lab4_Gayathri_chinka/
├── Lab4_Report.md              # Main submission report 
├── data.csv                    # Bankruptcy dataset (6,819 companies)
├── data_quality_check.py       # Data quality analysis script
├── psi_analysis.py            # Population Stability Index implementation
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🎯 Lab 4 Objectives

Research and justify decisions for 17 key areas in binary classification:
1. Model selection (3 models including benchmark)
2. Data preprocessing strategies  
3. Class imbalance handling
4. Outlier detection and treatment
5. Sampling bias assessment (PSI)
6. Feature scaling approaches
7. Evaluation metrics selection
8. Cross-validation strategies
9. And 9 additional decision areas...

---

## 🔬 Key Research Findings

### **Dataset Characteristics**
- **6,819 companies** with **96 financial features**
- **Severe class imbalance:** 96.8% non-bankrupt vs 3.2% bankrupt (1:29 ratio)
- **Zero missing values** (clean dataset)

### **Data Quality Issues**
- **13 features** with extreme outliers (billion-dollar impossible values)  
- **Current Ratio** max: 2.75 billion (normal range: 0.5-3.0)
- **Most ratios** properly normalized (0.0-1.0 range)

### **Statistical Validation**
- **PSI analysis:** All features <0.1 (no distribution drift)
- **Stratified sampling:** Maintains class balance across splits
- **Outlier analysis:** 4-9% outliers per feature (normal for financial data)

---

## 🚀 How to Run Analysis

### **Setup Environment**
```bash
# Navigate to project folder
cd lab4_Gayathri_chinka

# Activate virtual environment  
lab4_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Run Data Quality Analysis**
```bash
python data_quality_check.py
```
**Expected Output:** Dataset overview, outlier detection, class distribution analysis

### **Run PSI Analysis**
```bash
python psi_analysis.py
```
**Expected Output:** Train/test distribution comparison, PSI visualization

---

## 📊 Research Results Summary

| Analysis | Finding | Research Impact |
|----------|---------|----------------|
| **Class Balance** | 1:29 imbalance ratio | Justifies SMOTE oversampling |
| **Data Quality** | 13 features with impossible values | Supports outlier capping strategy |
| **Distribution Drift** | PSI <0.1 for all features | Validates train/test split approach |
| **Missing Data** | Zero missing values | Simplifies preprocessing pipeline |

---

## 📝 Main Deliverable

**`Lab4_Report.md`** - Complete research report addressing all 17 decision areas with:
- Model selection rationale (Logistic Regression + Random Forest + XGBoost)
- Data preprocessing strategies based on actual findings
- Class imbalance handling approach (SMOTE justified by 1:29 ratio)
- Statistical validation using PSI analysis
- All decisions supported by real data evidence

---

## 🎥 Video Presentation

**5-minute walkthrough** covering:
1. Data discovery (class imbalance, outliers)
2. Model selection rationale  
3. Preprocessing decisions
4. Validation strategies
5. How findings support research choices

**Focus:** Research reasoning, not code details. Avoid reading word-for-word.

---

## ✅ Lab 4 Success Criteria

- [x] 17 decision areas addressed with jot notes format
- [x] Real data analysis supporting all decisions  
- [x] PSI implementation demonstrating statistical understanding
- [x] Class imbalance quantified and addressed
- [x] Data quality issues identified and solutions proposed
- [x] Model selection justified by dataset characteristics
- [x] Concise explanations focused on "why" not "what"


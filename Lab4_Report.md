# My Lab 4 Project - Bankruptcy Prediction Research

Hey! This is my Lab 4 project where I had to research and justify modeling decisions for predicting company bankruptcies. Pretty interesting stuff once I dug into the data!

## 📂 What's in this folder

Here's what I've got set up:

- **`Lab4_Report.md`** - This is my main submission that answers all 17 research questions
- **`data.csv`** - The bankruptcy dataset (6,819 companies with tons of financial ratios)
- **`data_quality_check.py`** - Script I wrote to explore the data and find problems
- **`psi_analysis.py`** - My PSI implementation to check for distribution drift
- **`requirements.txt`** - All the Python packages I needed
- **`README.md`** - This file explaining everything

## 🎯 What I was supposed to do

My professor wanted me to research 17 different decision areas for binary classification and explain WHY I made each choice (not just what I chose). Had to use jot notes format or get a zero! 😅

The big areas were things like:
- Which 3 models to pick (needed a benchmark)
- How to handle class imbalance 
- Outlier treatment strategies
- Feature selection approaches
- Evaluation metrics that make sense
- And a bunch more...

## 🔍 Cool stuff I discovered

When I actually ran my analysis, I found some pretty wild things:

### The class imbalance is INSANE
- Only 220 bankrupt companies out of 6,819 total
- That's just 3.2% - way more imbalanced than I expected
- 1:29 ratio means I definitely need SMOTE or something similar

### The data has serious quality issues
- Found 13 features with completely impossible values
- Current Ratio had a max of 2.75 BILLION (normal range is like 0.5-3.0)
- Revenue Per Share hit 3.02 billion - clearly data entry errors
- But most of the other ratios looked pretty normal (0-1 range)

### My train/test split looks solid
- PSI values all under 0.1 for the features I tested
- No distribution drift between training and test data
- Stratified sampling worked well to maintain the bankruptcy rate

## 🚀 How to run my code

If you want to see my analysis:

1. **Get set up:**
   ```bash
   # Make sure you're in the right folder
   cd lab4_Gayathri_chinka
   
   # Turn on the virtual environment
   lab4_env\Scripts\activate
   
   # Install packages (should already be done)
   pip install -r requirements.txt
   ```

2. **Run the data quality check:**
   ```bash
   python data_quality_check.py
   ```
   This shows you the class imbalance and finds all those crazy outliers I mentioned.

3. **Run the PSI analysis:**
   ```bash
   python psi_analysis.py
   ```
   This creates the plot showing distribution stability between train/test sets.

## 📊 What the results look like

When you run `data_quality_check.py`, you'll see:
- Dataset shape: (6819, 96) 
- Missing values: 0 (nice!)
- A bunch of warnings about features with extreme values
- Class distribution showing the 96.8% vs 3.2% split
- Stats for key financial ratios

The PSI analysis gives you a nice bar chart where everything should be green (PSI < 0.1).

## 🎯 How this connects to my decisions

All this analysis actually drove my research decisions:

- **Severe imbalance** → SMOTE oversampling
- **Crazy outliers** → Tree models more robust than linear
- **Stable distributions** → Stratified CV approach works
- **Clean feature ratios** → Most preprocessing already done

Pretty cool how the real data backed up my theoretical research!

## 📝 For the video presentation

I'm planning to talk about:
1. The shocking class imbalance I discovered (1:29 ratio)
2. Those impossible billion-dollar financial ratios
3. How PSI analysis confirmed my sampling approach 
4. Why this led me to pick robust models like Random Forest and XGBoost

The key is showing how my actual findings support the decisions, not just theoretical reasoning.

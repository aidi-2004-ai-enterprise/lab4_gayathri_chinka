# My Lab 4 Project - Bankruptcy Prediction Research

Hey! I'm Gayathri and this is my Lab 4 project where I had to dive deep into company bankruptcy prediction. Honestly, it turned out way more interesting than I expected once I started digging into the actual data!

## 📂 What's in my project folder

Here's everything I've put together:

- **`Lab4_Report.md`** - My main submission answering all 17 research questions (this is what gets graded!)
- **`data.csv`** - The bankruptcy dataset with 6,819 companies and tons of financial ratios
- **`data_quality_check.py`** - Script I wrote to explore the data and hunt for problems
- **`psi_analysis.py`** - My PSI implementation to check if my train/test split is actually valid
- **`psi_analysis_results.png`** - Chart showing my PSI results (spoiler: all green bars!)
- **`data_quality_overview.png`** - Screenshot of the crazy outliers I found
- **`data_quality_summary.png`** - Summary of my key findings
- **`requirements.txt`** - All the Python packages I needed to make this work
- **`README.md`** - This file where I'm explaining everything

## 🎯 What I was supposed to accomplish

My professor gave us this assignment where we had to research 17 different decision areas for binary classification and justify WHY we made each choice. The catch? Had to use jot notes format or we'd get a zero! 😅 No paragraphs allowed.

The big research areas included stuff like:
- Picking 3 models (including a benchmark)
- Figuring out how to handle class imbalance
- Deciding on outlier treatment strategies  
- Choosing the right evaluation metrics
- Setting up proper cross-validation
- And a whole bunch more technical decisions...

## 🔍 Wild stuff I discovered when I analyzed the data

When I actually ran my analysis scripts, I found some pretty shocking things:

### The class imbalance is absolutely insane
- Only 220 bankrupt companies out of 6,819 total companies
- That's literally just 3.2% bankruptcy rate - way worse than I expected
- We're talking about a 1:29 imbalance ratio here
- This immediately told me I'd need SMOTE or some serious oversampling technique

### The data has major quality problems
- I found 13 features with completely impossible values
- Current Ratio had a maximum of 2.75 BILLION (normal range should be like 0.5 to 3.0)
- Revenue Per Share hit 3.02 billion - clearly these are data entry errors
- But most of the other financial ratios looked pretty reasonable (0-1 range)

### My train/test split actually looks solid though
- All my PSI values came back under 0.1 for the features I tested
- No distribution drift between my training and test data
- Stratified sampling worked perfectly to maintain the bankruptcy rate

## 🚀 How to run my code (if you want to replicate this)

If you're curious to see my analysis in action:

1. **Get everything set up:**
   ```bash
   # Navigate to my project folder
   cd "C:\Users\sindh\OneDrive\Desktop\Assingment AI\AIDI2004_Labs\lab4_Gayathri_chinka"
   
   # Turn on my virtual environment
   C:\Users\sindh\lab4_Gayathri_chinka\lab4_env\Scripts\activate
   
   # Install all the packages (should already be done)
   pip install -r requirements.txt
   ```

2. **Run my data quality analysis:**
   ```bash
   python data_quality_check.py
   ```
   This will show you the class imbalance and all those crazy outliers I mentioned.

3. **Run my PSI analysis:**
   ```bash
   python psi_analysis.py
   ```
   This creates a nice chart showing distribution stability between train/test sets.

## 📊 What you'll see when you run my scripts

When you run `data_quality_check.py`, you'll get:
- Dataset shape: (6819, 96) - pretty big dataset!
- Missing values: 0 (surprisingly clean)
- A bunch of warnings about features with extreme values
- Class distribution showing that brutal 96.8% vs 3.2% split
- Basic stats for the key financial ratios

The PSI analysis spits out a nice bar chart where everything should be green (PSI < 0.1).

## 📸 Visual proof of my findings

I captured screenshots of my key results because the professor wanted evidence:
- **PSI Chart:** Beautiful green bars showing no distribution drift
- **Data Quality Output:** Terminal screenshots showing all those billion-dollar outliers
- **Class Imbalance Visualization:** That stark 1:29 ratio display

Having these screenshots proves I actually did the analysis instead of just making stuff up!

## 🎯 How my real findings drove my research decisions

This is the cool part - all my actual data analysis backed up my theoretical choices:

- **Severe 1:29 imbalance** → Had to go with SMOTE oversampling
- **Crazy billion-dollar outliers** → Tree models way more robust than linear models
- **Stable PSI distributions** → My stratified CV approach actually works
- **Most ratios well-behaved** → Don't need tons of preprocessing for most features

## 📊 My research findings summary

Here's a quick breakdown of what I discovered and how it impacted my decisions:

| What I Analyzed | What I Found | How This Changed My Approach |
|----------------|--------------|----------------------------|
| **Class Balance** | 1:29 imbalance ratio (only 3.2% bankrupt) | Had to use SMOTE oversampling instead of just class weights |
| **Data Quality** | 13 features with impossible billion-dollar values | Chose tree models over linear (more robust to outliers) |
| **Distribution Drift** | PSI <0.1 for all features I tested | Proved my stratified sampling approach actually works |
| **Missing Data** | Zero missing values across entire dataset | Simplified my preprocessing pipeline significantly |
| **Feature Ranges** | Most ratios properly normalized (0-1 range) | Don't need extensive scaling for most features |
| **Outlier Patterns** | 4-9% outliers per feature (normal for finance) | Keep legitimate extremes, only cap impossible values |

This table basically shows how my actual data exploration drove every single modeling decision I made!




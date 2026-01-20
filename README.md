# House Price Prediction with Gradient Boosting

A robust real estate valuation system leveraging machine learning to estimate property values based on key architectural and spatial features. Built with Gradient Boosting and feature engineering best practices.

---

## Demo Video

[![Watch the video](https://img.youtube.com/vi/SEAnfUnSGbU/hqdefault.jpg)](https://www.youtube.com/watch?v=SEAnfUnSGbU)

---
## Problem Statement

Accurate home valuation is critical for buyers, sellers, and lenders in dynamic real estate markets. Traditional appraisal methods can be subjective and time-consuming. This model automates price estimation using:

- Historical sales data
- Property characteristics
- Non-linear relationships between features
- Market trends

---

## Key Features

- Outlier-resistant data preprocessing (2nd-98th percentile filtering)
- Log-transformed target variable for normalized distribution
- 8 optimized features capturing property essence:
  - Living area
  - Construction quality
  - Garage capacity
  - Basement size
  - Year built
  - Bedrooms
  - Bathrooms (full/half)
- Robust pipeline architecture (Scaler + Model)

---

## Model Architecture

```python
GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
) 
```
---

# Setup Instructions

## 1) Clone the Repository
```bash

git clone https://github.com/jatin-wig/spam-detector-large-scale.git
```

## 2) Install Dependencies
```bash
pip install -r requirements.txt
```

## 3) Run the App
```bash
streamlit run app.py
 ```
or 
```bash
python -m streamlit run app.py 
```
--- 

## Demo

You can access the live demo of the application by visiting the following link:  
[View Demo](https://house-prediction-model-jatin-wig.streamlit.app/)

# Built by Jatin Wig
### GitHub: https://github.com/jatin-wig

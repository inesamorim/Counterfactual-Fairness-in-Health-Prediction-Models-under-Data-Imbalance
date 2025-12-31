# Counterfactual Fairness under Gender Imbalance: An Empirical Study on Health Decision Models
"AI and Society" Course Individual Project - FCUP (2025/2026)

This project investigates counterfactual fairness in machine learning models for health decision-making, focusing on gender imbalance. Using the Cleveland Heart Disease dataset, the study evaluates how different training strategies (train-test split, stratification, weighted training, SMOTE) affect model predictions, fairness metrics, and counterfactual explanations.

---
## Features
- Analysis of gender-specific performance and fairness metrics.
- Counterfactual explanations for individual-level fairness evaluation using DiCE.
- Comparison of multiple imbalance mitigation strategies (stratification, weighting, SMOTE).
- Heatmap visualizations of feature changes in counterfactuals.
- Modular utility functions for metrics, model evaluation, and visualization.
---

## Instalation / Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage
How to run the project or reproduce results.

1. Clone the repository:
```bash
git clone https://github.com/inesamorim/Counterfactual-Fairness-in-Health-Prediction-Models-under-Data-Imbalance.git
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```

---


## Project Structure

```bash
├─ main.ipynb                  # Notebook with full workflow
├─ utils/                      # Utility modules
│   ├─ __pycache__/            
│   ├─ __init__.py
│   ├─ counterfactuals.py      # Functions to generate counterfactuals
│   ├─ metrics.py              # Custom fairness and performance metrics
│   ├─ model_evaluation.py     # Functions for training and evaluating models
│   └─ visualization.py        # Plotting and figure utilities
├─ datasets/                   # Raw and processed datasets
│   ├─ heart_disease_preprocessed.csv           
│   ├─ heart_disease.csv                
├─ requirements.txt            # Project dependencies
└─ report.tex                  # LaTeX report

```

## Results Summary

- Male predictions rely mostly on physiological features.

- Female predictions often require sex flips under imbalance.

- SMOTE improves group-level fairness but increases individual-level sensitivity.

- Stratified splitting stabilizes counterfactual explanations.
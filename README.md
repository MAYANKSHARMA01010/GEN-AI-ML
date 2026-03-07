# 📊 Telco Customer Churn Prediction: An End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

This repository contains a professional, end-to-end Machine Learning pipeline designed to predict whether a telecom customer will churn (cancel their service) based on their account information.

This project serves as both a **Functional Machine Learning Application** and a **Comprehensive Study Guide** for understanding the mathematical theory behind classical algorithms.

---

## 🌟 Key Features

- **End-to-End Pipeline:** From raw data cleaning to model deployment readiness.
- **In-Depth Mathematical Theory:** Detailed explanations of the algorithms and evaluation metrics.
- **Dual Data Scaling:** Compares model performance across MinMax scaled and Standard (Z-Score) scaled datasets.
- **Model Showdown:** Evaluates Linear Regression (thresholded), Logistic Regression, and Decision Trees to find the optimal solution.

---

## 📁 Repository Structure

### **Phase 1: Data Mastery & Preprocessing**

- **`1_Data_Mastery_&_Preprocessing.ipynb`**
  - **Goal:** Transform raw, messy data into pristine numbers that algorithms can understand.
  - **Highlights:** Handles missing values (`TotalCharges`), caps extreme outliers, performs Exploratory Data Analysis (EDA), and uses Pandas `get_dummies()` to encode categorical text data into binary features.
  - **Dual Pipelines:** This notebook explicitly exports TWO separate datasets for testing:
    1. **MinMax Scaled (`Telco-Customer-Churn-MinMax.csv`):** Numbers squished strictly between 0 and 1.
    2. **Standard Scaled ($Z$-Score) (`Telco-Customer-Churn-Standard.csv`):** Centered numbers with a mean of 0 and Standard Deviation of 1.

### **Phase 2: Machine Learning Models (The "Big Three")**

These notebooks evaluate three fundamentally different "brains" on both versions of the scaled data, complete with deeply documented mathematical explanations.

- **`2.1_Modeling_LinearRegression.ipynb`**
  - Uses a flat Mathematical Line ($y = mx + b$).
  - Demonstrates how to "hack" Regression for Classification by setting a distinct $0.5$ hard threshold.
- **`2.2_Modeling_LogisticRegression.ipynb`**
  - The industry standard for Binary Classification.
  - Uses the S-Shaped **Sigmoid Curve** to force outputs precisely between 0 and 1, granting us a literal _probability_ of churn rather than an arbitrary number.
- **`2.3_Modeling_DecisionTree.ipynb`**
  - A hierarchical, non-linear "Flowchart" model.
  - Includes a deep-dive into the mathematical **Gini Impurity** Index ($1 - \sum (p_i)^2$) and exactly how trees evaluate node purity down to 0.0.

### **Phase 3: The Ultimate Winner**

- **`3_Model_Comparison.ipynb`**
  - Retrains all 3 models blindly, extracts their final **Accuracy** and **F1-Scores**, and plots them in a unified Seaborn Grouped Bar Chart.
  - Demonstrates the mathematical flaws of unrestrained Decision Trees (Overfitting) and declares the definitive algorithm for production deployment.

---

## 📐 The Evaluation Metrics Masterclass

This repository doesn't just call `model.fit()`. Throughout the notebooks, you will find extensive Markdown theory defining exactly how the machine judges "Error":

- **Regression Metrics:** Formula breakdowns for $R^2$, SSR (Sum of Squared Residuals), SST (Total Sum of Squares), MAE, MSE, and RMSE.
- **Classification Metrics:** The anatomy of the **Confusion Matrix** (True Positives, False Positives/Type I Errors, False Negatives/Type II Errors) and the subsequent mathematical formulas for **Accuracy**, **Precision**, **Recall**, and the **F1-Score**.
- **Visual Data:** Dynamic Matplotlib code blocks that plot the Sigmoid Curve, Regression Lines, and actual graphical representations of the trained models.

---

## 💻 How to Run Locally

### Prerequisites

Ensure you have Python 3.8+ installed along with the following libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn notebook
```

### Execution Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/GEN-AI-ML.git
   cd GEN-AI-ML
   ```

2. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

3. **Run the Notebooks Sequentially:**
   - ➜ Run `1_Data_Mastery_&_Preprocessing.ipynb` first to generate the necessary scaled `.csv` files.
   - ➜ Run the **Phase 2** modeling notebooks in any order to train the individual models and explore the theory.
   - ➜ Run `3_Model_Comparison.ipynb` to execute the final showdown and compare the results!

---

## 📊 Dataset

The dataset used is the **Telco Customer Churn** dataset. It contains information about a fictional telco company that provided home phone and Internet services to thousands of customers, indicating which customers have left, stayed, or signed up for their service.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

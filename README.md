# Customer Churn Prediction

This project builds a machine learning model to predict customer churn based on various features such as demographics, account information, and transaction history.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Enhancements](#future-enhancements)

## Overview
Customer churn prediction helps businesses identify customers who are likely to leave, enabling proactive retention strategies. This project utilizes **machine learning** techniques to analyze customer behavior and predict churn.

## Dataset
The dataset contains customer information, including:
- **Demographic Details** (Age, Gender, Geography, etc.)
- **Account Information** (Balance, Tenure, Number of Products)
- **Transaction Behavior** (Credit Score, Estimated Salary, etc.)
- **Target Variable**: `Exited` (1 = Customer Churned, 0 = Retained)

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras
- **Machine Learning Models:** Logistic Regression, Neural Networks (ANN with TensorFlow)

## Modeling Approach
1. **Data Preprocessing:**
   - Handling missing values, encoding categorical features (`Geography`, `Gender`)
   - Feature scaling using **StandardScaler**
2. **Model Training:**
   - Split data into training (80%) and testing (20%)
   - Used different classifiers including **Random Forest, XGBoost, and ANN**
   - Evaluated performance using **accuracy, precision, recall, and AUC-ROC**
3. **Neural Network Implementation:**
   - **Input Layer:** Features from the dataset
   - **Hidden Layers:** Fully connected dense layers with ReLU activation
   - **Output Layer:** Sigmoid activation for binary classification
   - **Loss Function:** `binary_crossentropy`
   - **Optimizer:** Adam

## Results
- Achieved **high accuracy** and **AUC-ROC score** indicating strong model performance
- The **ANN model outperformed traditional models** in predicting churn

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Execute `customer_churn_prediction.ipynb` to train and evaluate the model

## Future Enhancements
- Deploy model using **Flask or FastAPI**
- Improve feature engineering for better predictions
- Experiment with **LSTM models** for time-series customer behavior analysis


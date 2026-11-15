# Bike Sharing Demand Analysis and Prediction

## 1. Project Overview
This project focuses on analyzing bike-sharing rental
patterns and predicting hourly bike rental demand using
Machine Learning techniques.

## 2. Objectives
- Analyze hourly bike rental demand.
- Identify rental patterns based on time and weather.
- Clean and preprocess the dataset.
- Perform Exploratory Data Analysis (EDA).
- Develop and evaluate Machine Learning models.
- Predict hourly bike rental demand.

## 3. Dataset Description
Dataset: Bike Sharing Dataset - Hourly Data

Source: UCI Machine Learning Repository

Dataset file: data set/hour_messy.csv

Target variable: cnt (Total bike rentals)

## 4. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 5. Methodology
1. Data Loading
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Train-Test Splitting
6. Model Training
7. Model Evaluation
8. Demand Prediction

## 6. Data Preprocessing
- Handled missing values.
- Removed duplicate records.
- Handled abnormal normalized values.
- Converted date column into datetime format.
- Created cyclic time features.
- Corrected 15 inconsistent total rental values
  using casual + registered, assuming those two
  component columns were correct.

## 7. Machine Learning
Models explored:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Final model: Random Forest Regressor

## 8. Final Model Performance
- MAE: 47.40
- RMSE: 72.66
- R2 Score: 0.8913

Training records: 13,891
Testing records: 3,488

## 9. Project Outputs
- Actual vs Predicted graph
- Residual Plot
- Residual Distribution
- Feature Importance graph
- Prediction results CSV
- Trained Random Forest model

## 10. How to Run the Project

Install the required libraries:

pip install -r requirements.txt

Run the Python program:

python main.py

## 11. Conclusion
The project demonstrates how data preprocessing,
exploratory analysis and machine learning can be used
to analyze and predict bike-sharing demand.
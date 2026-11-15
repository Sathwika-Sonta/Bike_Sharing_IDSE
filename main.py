import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Load our messy dataset
data = pd.read_csv("data set/hour_messy.csv")

print("========== DATASET OVERVIEW ==========")

# 1. Dataset dimensions
print("\nDataset shape:")
print(data.shape)

# 2. Column names
print("\nColumn names:")
print(data.columns.tolist())

# 3. Missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# 4. Total missing values
print("\nTotal missing values:")
print(data.isnull().sum().sum())

# 5. Duplicate records
print("\nNumber of duplicate rows:")
print(data.duplicated().sum())

# 6. Data types
print("\nData types:")
print(data.dtypes)

# 7. Statistical summary
print("\nStatistical summary:")
print(data.describe())

# 8. Check rental count consistency
print("\nIncorrect rental totals:")
print((data["casual"] + data["registered"] != data["cnt"]).sum())

print("\n========== INSPECTION COMPLETED ==========")
print("\n========== CHECKING ABNORMAL VALUES ==========")

# Temperature outside 0 to 1
print("\nAbnormal temperature records:")
print(data[(data["temp"] < 0) | (data["temp"] > 1)][["temp"]])

# Humidity outside 0 to 1
print("\nAbnormal humidity records:")
print(data[(data["hum"] < 0) | (data["hum"] > 1)][["hum"]])

# Windspeed outside 0 to 1
print("\nAbnormal windspeed records:")
print(data[(data["windspeed"] < 0) | (data["windspeed"] > 1)][["windspeed"]])

# Incorrect rental totals
print("\nRecords with incorrect rental totals:")
print(data[data["casual"] + data["registered"] != data["cnt"]][
    ["casual", "registered", "cnt"]
])
# STEP 3: REMOVE DUPLICATE RECORDS

print("\n========== DUPLICATE REMOVAL ==========")

# Create a separate working dataset
clean_data = data.copy()

# Count duplicates before removal
print("Rows before removing duplicates:", len(clean_data))

# Remove exact duplicate rows
clean_data = clean_data.drop_duplicates()

# Check duplicates after removal
print("Rows after removing duplicates:", len(clean_data))

print("Remaining duplicate rows:", clean_data.duplicated().sum())
# STEP 4: HANDLE ABNORMAL VALUES

print("\n========== HANDLING ABNORMAL VALUES ==========")

# Replace invalid temperature values
clean_data.loc[
    (clean_data["temp"] < 0) | (clean_data["temp"] > 1), "temp"
] = float("nan")

# Replace invalid humidity values
clean_data.loc[
    (clean_data["hum"] < 0) | (clean_data["hum"] > 1), "hum"
] = float("nan")

# Replace invalid windspeed values
clean_data.loc[
    (clean_data["windspeed"] < 0) | (clean_data["windspeed"] > 1), "windspeed"
] = float("nan")

# Check abnormal values after treatment
print("Abnormal temperature:", 
      ((clean_data["temp"] < 0) | (clean_data["temp"] > 1)).sum())

print("Abnormal humidity:", 
      ((clean_data["hum"] < 0) | (clean_data["hum"] > 1)).sum())

print("Abnormal windspeed:", 
      ((clean_data["windspeed"] < 0) | (clean_data["windspeed"] > 1)).sum())

print("\nAbnormal values converted into missing values.")
# STEP 5A: CHECK UPDATED MISSING VALUES

print("\n========== UPDATED MISSING VALUES ==========")

print("\nMissing values in each column:")
print(clean_data.isnull().sum())

print("\nTotal missing values:")
print(clean_data.isnull().sum().sum())
# STEP 5B: HANDLE MISSING VALUES

print("\n========== MISSING VALUE IMPUTATION ==========")

# Fill numerical columns using median
numeric_columns = ["temp", "atemp", "hum", "windspeed"]

for col in numeric_columns:
    median_value = clean_data[col].median()
    clean_data[col] = clean_data[col].fillna(median_value)

# Fill weather situation using mode
weather_mode = clean_data["weathersit"].mode()[0]
clean_data["weathersit"] = clean_data["weathersit"].fillna(weather_mode)

# Verify missing values after imputation
print("\nMissing values after imputation:")
print(clean_data.isnull().sum())

print("\nTotal missing values after imputation:")
print(clean_data.isnull().sum().sum())
# STEP 6: DATE COLUMN CONVERSION

print("\n========== DATE COLUMN CONVERSION ==========")

# Convert dteday into datetime format
clean_data["dteday"] = pd.to_datetime(
    clean_data["dteday"], format="mixed", errors="coerce"
)

# Check datatype after conversion
print("\nData type of dteday:")
print(clean_data["dteday"].dtype)

# Check invalid dates
print("\nInvalid dates:")
print(clean_data["dteday"].isnull().sum())

# Display first five dates
print("\nFirst five dates:")
print(clean_data["dteday"].head())
# STEP 7A: CHECK DATA TYPES

print("\n========== DATA TYPE INSPECTION ==========")

print("\nCurrent data types:")
print(clean_data.dtypes)

print("\nDataset information:")
clean_data.info()
# STEP 7B: CORRECT WEATHER DATA TYPE

clean_data["weathersit"] = clean_data["weathersit"].astype(int)

print("\nCorrected data types:")
print(clean_data.dtypes)

print("\nUnique weather categories:")
print(sorted(clean_data["weathersit"].unique()))

# Check whether any missing values remain
print("\nTotal missing values:", clean_data.isnull().sum().sum())
# STEP 8: CHECK RENTAL COUNT CONSISTENCY

print("\n========== RENTAL COUNT VALIDATION ==========")

# Find records where rental totals do not match
mismatch = clean_data[
    clean_data["casual"] + clean_data["registered"] != clean_data["cnt"]
]

print("\nNumber of inconsistent records:")
print(len(mismatch))

print("\nInconsistent records:")
print(mismatch[
    ["instant", "dteday", "hr", "casual", "registered", "cnt"]
])

# Check rental count relationship
print("\nTotal inconsistent records:",
      (clean_data["casual"] + clean_data["registered"] != clean_data["cnt"]).sum())
# STEP 9: CORRECT RENTAL COUNT INCONSISTENCIES

print("\n========== CORRECTING RENTAL COUNTS ==========")

# Recalculate total rentals
clean_data["cnt"] = clean_data["casual"] + clean_data["registered"]

# Verify consistency
remaining_mismatches = (
    clean_data["casual"] + clean_data["registered"] != clean_data["cnt"]
).sum()

print("Remaining inconsistent records:", remaining_mismatches)

print("\nRental count correction completed.")
# STEP 10: SAVE CLEANED DATASET

print("\n========== SAVING CLEANED DATASET ==========")

clean_data.to_csv("bike_sharing_cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")

print("\nFinal dataset shape:")
print(clean_data.shape)

print("\nFinal missing values:")
print(clean_data.isnull().sum().sum())

print("\nFinal duplicate records:")
print(clean_data.duplicated().sum())
# STEP 11: EXPLORATORY DATA ANALYSIS

import matplotlib.pyplot as plt
import seaborn as sns

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Statistical summary
print("\nStatistical summary:")
print(clean_data[["temp", "atemp", "hum", "windspeed", "cnt"]].describe())

# 2. Average bike rentals
print("\nAverage bike rentals:")
print(clean_data["cnt"].mean())

# 3. Maximum bike rentals
print("\nMaximum bike rentals:")
print(clean_data["cnt"].max())

# 4. Minimum bike rentals
print("\nMinimum bike rentals:")
print(clean_data["cnt"].min())

# 5. Average rentals by hour
print("\nAverage rentals by hour:")
print(clean_data.groupby("hr")["cnt"].mean())
# STEP 12: DATA VISUALIZATION

# Arrange data by date and hour
clean_data = clean_data.sort_values(
    by=["dteday", "hr"]
)

plt.figure(figsize=(14, 10))

# Graph 1: Hourly bike rentals
plt.subplot(2, 2, 1)

hourly_data = clean_data.groupby("hr")["cnt"].mean()

plt.plot(hourly_data.index, hourly_data.values, marker="o")
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Rentals")
plt.grid(True)

# Graph 2: Monthly bike rentals
plt.subplot(2, 2, 2)

monthly_data = clean_data.groupby("mnth")["cnt"].mean()

plt.bar(monthly_data.index, monthly_data.values)
plt.title("Average Bike Rentals by Month")
plt.xlabel("Month")
plt.ylabel("Average Rentals")

# Graph 3: Weather vs bike rentals
plt.subplot(2, 2, 3)

weather_data = clean_data.groupby("weathersit")["cnt"].mean()

plt.bar(weather_data.index, weather_data.values)
plt.title("Weather vs Bike Rentals")
plt.xlabel("Weather Situation")
plt.ylabel("Average Rentals")

# Graph 4: Working day vs non-working day
plt.subplot(2, 2, 4)

working_data = clean_data.groupby("workingday")["cnt"].mean()

plt.bar(["Non-working day", "Working day"], working_data.values)
plt.title("Working Day vs Non-working Day Bike Rentals")
plt.xlabel("Day Type")
plt.ylabel("Average Rentals")

plt.tight_layout()
plt.show()
# STEP 13: CORRELATION ANALYSIS

print("\n========== CORRELATION ANALYSIS ==========")

# Select important numerical columns
correlation_columns = [
    "hr", "temp", "atemp", "hum", "windspeed", "cnt"
]

# Calculate correlation
correlation_matrix = clean_data[correlation_columns].corr()

print("\nCorrelation with bike rental count:")
print(correlation_matrix["cnt"].sort_values(ascending=False))

# Plot correlation heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Bike Sharing Dataset")
plt.tight_layout()
plt.show()
# STEP 14: BIKE RENTAL DISTRIBUTION

print("\n========== BIKE RENTAL DISTRIBUTION ==========")

plt.figure(figsize=(10, 5))

sns.histplot(
    clean_data["cnt"],
    bins=40,
    kde=True
)

plt.title("Distribution of Bike Rental Counts")
plt.xlabel("Number of Bike Rentals")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
# STEP 15: FEATURE ENGINEERING

import numpy as np

print("\n========== FEATURE ENGINEERING ==========")

# Create cyclic hour features
clean_data["hour_sin"] = np.sin(2 * np.pi * clean_data["hr"] / 24)
clean_data["hour_cos"] = np.cos(2 * np.pi * clean_data["hr"] / 24)

# Create cyclic month features
clean_data["month_sin"] = np.sin(2 * np.pi * clean_data["mnth"] / 12)
clean_data["month_cos"] = np.cos(2 * np.pi * clean_data["mnth"] / 12)

# Display new features
print("\nNewly created features:")
print(clean_data[
    ["hr", "hour_sin", "hour_cos",
     "mnth", "month_sin", "month_cos"]
].head())

print("\nUpdated dataset shape:")
print(clean_data.shape)
# STEP 16: PREPARE INPUT AND TARGET VARIABLES

print("\n========== INPUT AND TARGET PREPARATION ==========")

# Sort records chronologically
clean_data = clean_data.sort_values(
    by=["dteday", "hr"]
)

# Separate input features
X = clean_data.drop(
    columns=["cnt", "casual", "registered", "instant", "dteday"]
)

# Separate target variable
y = clean_data["cnt"]

print("\nInput features:")
print(X.columns.tolist())

print("\nInput shape:", X.shape)
print("Target shape:", y.shape)


# STEP 17: TRAIN-TEST SPLITTING

print("\n========== TRAIN-TEST SPLITTING ==========")

# Calculate 80% training size
split_index = int(len(clean_data) * 0.80)

# Split chronologically
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nTraining period:")
print(clean_data["dteday"].iloc[0],
      "to",
      clean_data["dteday"].iloc[split_index - 1])

print("\nTesting period:")
print(clean_data["dteday"].iloc[split_index],
      "to",
      clean_data["dteday"].iloc[-1])
# STEP 16 AND 17: FINAL FEATURE PREPARATION AND DATE-BASED SPLIT

print("\n========== FINAL DATA PREPARATION ==========")

# Correct cyclic month features
clean_data["month_sin"] = np.sin(
    2 * np.pi * (clean_data["mnth"] - 1) / 12
)

clean_data["month_cos"] = np.cos(
    2 * np.pi * (clean_data["mnth"] - 1) / 12
)

# Sort data chronologically
clean_data = clean_data.sort_values(
    by=["dteday", "hr"]
).reset_index(drop=True)

# Prepare input and target
X = clean_data.drop(
    columns=["cnt", "casual", "registered", "instant", "dteday"]
)

y = clean_data["cnt"]

# Split using complete dates
unique_dates = sorted(clean_data["dteday"].unique())

split_index = int(len(unique_dates) * 0.80)

split_date = unique_dates[split_index]

train_mask = clean_data["dteday"] < split_date
test_mask = clean_data["dteday"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining period:")
print(clean_data.loc[train_mask, "dteday"].min(),
      "to",
      clean_data.loc[train_mask, "dteday"].max())

print("\nTesting period:")
print(clean_data.loc[test_mask, "dteday"].min(),
      "to",
      clean_data.loc[test_mask, "dteday"].max())

print("\nTraining and testing dates overlap:",
      not set(clean_data.loc[train_mask, "dteday"]).isdisjoint(
          set(clean_data.loc[test_mask, "dteday"])
      ))
# STEP 18: MACHINE LEARNING MODEL TRAINING

print("\n========== MODEL TRAINING ==========")

# MODEL 1: LINEAR REGRESSION
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

linear_model.fit(X_train, y_train)

print("\nLinear Regression training completed!")


# MODEL 2: DECISION TREE
tree_model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

print("Decision Tree training completed!")


# MODEL 3: RANDOM FOREST
forest_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

forest_model.fit(X_train, y_train)

print("Random Forest training completed!")


print("\nAll three models trained successfully!")
# STEP 19: MODEL EVALUATION

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import numpy as np

print("\n========== MODEL EVALUATION ==========")

# Store models
models = {
    "Linear Regression": linear_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}

results = []

predictions = {}

# Evaluate each model
for name, model in models.items():

    # Predict bike rentals
    y_pred = model.predict(X_test)

    predictions[name] = y_pred

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

# Create comparison table
results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON:")
print(results_df.to_string(index=False))

# Identify best model by RMSE
best_model = results_df.loc[
    results_df["RMSE"].idxmin(), "Model"
]

print("\nBest model based on RMSE:", best_model)
# STEP 20: MODEL PERFORMANCE VISUALIZATION

print("\n========== MODEL PERFORMANCE GRAPHS ==========")

# GRAPH 1: Actual vs Predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions["Random Forest"], alpha=0.4)

# Perfect prediction reference line
minimum = min(y_test.min(), predictions["Random Forest"].min())
maximum = max(y_test.max(), predictions["Random Forest"].max())

plt.plot([minimum, maximum], [minimum, maximum], "r--")

plt.title("Actual vs Predicted Bike Rentals - Random Forest")
plt.xlabel("Actual Rentals")
plt.ylabel("Predicted Rentals")
plt.grid(True)
plt.tight_layout()
plt.show()


# GRAPH 2: Compare RMSE of all models
plt.figure(figsize=(8, 5))

sns.barplot(data=results_df, x="Model", y="RMSE")

plt.title("RMSE Comparison of Machine Learning Models")
plt.ylabel("RMSE (Lower is Better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# GRAPH 3: Compare R2 scores
plt.figure(figsize=(8, 5))

sns.barplot(data=results_df, x="Model", y="R2 Score")

plt.title("R2 Score Comparison of Machine Learning Models")
plt.ylabel("R2 Score (Higher is Better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
# STEP 21: FEATURE IMPORTANCE ANALYSIS

print("\n========== FEATURE IMPORTANCE ==========")

importance = forest_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature importance ranking:")
print(importance_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
# STEP 22: BASELINE MODEL

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

baseline_model = DummyRegressor(strategy="mean")

# Train baseline model
baseline_model.fit(X_train, y_train)

# Predict test data
baseline_predictions = baseline_model.predict(X_test)

# Calculate evaluation metrics
baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
baseline_r2 = r2_score(y_test, baseline_predictions)

print("\n========== BASELINE MODEL RESULTS ==========")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2 Score:", baseline_r2)
# STEP 23: RESIDUAL ANALYSIS

residuals = y_test - predictions["Random Forest"]

plt.figure(figsize=(8, 5))
plt.scatter(predictions["Random Forest"], residuals, alpha=0.4)
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted Rental Count")
plt.ylabel("Residuals")
plt.title("Residual Analysis - Random Forest")
plt.grid(True)
plt.show()
# STEP 24: SAVE BEST MODEL

import joblib

joblib.dump(forest_model, "bike_sharing_random_forest.pkl")

print("Random Forest model saved successfully!")
# STEP 25: SAVE MODEL COMPARISON RESULTS

results_df.to_csv("model_comparison_results.csv", index=False)

print("Model comparison results saved successfully!")
# LOAD THE SAVED RANDOM FOREST MODEL

import joblib

loaded_model = joblib.load("bike_sharing_random_forest.pkl")

print("\n========== SAVED MODEL LOADED ==========")

print("Model loaded successfully!")
print(loaded_model)
# TESTING SAVED MODEL WITH ONE SAMPLE

sample_data = X_test.iloc[[0]]

actual_value = y_test.iloc[0]

predicted_value = loaded_model.predict(sample_data)[0]

print("\n========== SAMPLE PREDICTION ==========")

print("Actual Bike Rentals:", actual_value)

print("Predicted Bike Rentals:", round(predicted_value))

print("Prediction Error:", round(abs(actual_value - predicted_value), 2))
# PREDICTING FIRST 10 TEST SAMPLES

sample_predictions = loaded_model.predict(X_test.iloc[:10])

prediction_results = pd.DataFrame({
    "Actual Rentals": y_test.iloc[:10].values,
    "Predicted Rentals": sample_predictions.round().astype(int)
})

prediction_results["Prediction Error"] = abs(
    prediction_results["Actual Rentals"] -
    prediction_results["Predicted Rentals"]
)

print("\n========== FIRST 10 PREDICTIONS ==========")

print(prediction_results)
# SAVE RANDOM FOREST PREDICTION RESULTS

all_predictions = loaded_model.predict(X_test)

final_results = pd.DataFrame({
    "Actual Rentals": y_test.values,
    "Predicted Rentals": all_predictions.round().astype(int)
})

final_results["Prediction Error"] = abs(
    final_results["Actual Rentals"] -
    final_results["Predicted Rentals"]
)

final_results.to_csv(
    "random_forest_test_predictions.csv",
    index=False
)

print("\n========== PREDICTION RESULTS SAVED ==========")

print(final_results.head(10))

print("\nTotal predictions:", len(final_results))

print("Prediction results saved successfully!")
# RESIDUAL ANALYSIS

residuals = y_test.values - all_predictions

plt.figure(figsize=(8, 6))

plt.scatter(all_predictions, residuals, alpha=0.4)

plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted Rentals")
plt.ylabel("Residuals")

plt.title("Random Forest Residual Plot")

plt.grid(True)

plt.show()
# RESIDUAL DISTRIBUTION ANALYSIS

plt.figure(figsize=(8, 6))

plt.hist(residuals, bins=40, edgecolor="black")

plt.axvline(x=0, color="red", linestyle="--")

plt.xlabel("Residuals")
plt.ylabel("Frequency")

plt.title("Distribution of Random Forest Residuals")

plt.grid(True)

plt.show()


# RESIDUAL STATISTICS

print("\n========== RESIDUAL STATISTICS ==========")

print("Mean Residual:", round(residuals.mean(), 2))

print("Standard Deviation:", round(residuals.std(), 2))

print("Minimum Residual:", round(residuals.min(), 2))

print("Maximum Residual:", round(residuals.max(), 2))

print("Errors within 50 rentals:",
      round((abs(residuals) <= 50).mean() * 100, 2), "%")

print("Errors within 100 rentals:",
      round((abs(residuals) <= 100).mean() * 100, 2), "%")
# FINAL MODEL PERFORMANCE COMPARISON

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Random Forest predictions
rf_predictions = loaded_model.predict(X_test)

# Baseline predictions: training target mean
baseline_predictions = np.full(len(y_test), y_train.mean())

# Calculate metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
baseline_r2 = r2_score(y_test, baseline_predictions)

# Comparison table
final_comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Baseline Model": [baseline_mae, baseline_rmse, baseline_r2],
    "Random Forest": [rf_mae, rf_rmse, rf_r2]
})

print("\n========== FINAL MODEL COMPARISON ==========")

print(final_comparison.round(3).to_string(index=False))

# Improvement calculation
mae_improvement = ((baseline_mae - rf_mae) / baseline_mae) * 100
rmse_improvement = ((baseline_rmse - rf_rmse) / baseline_rmse) * 100

print("\nMAE Improvement:", round(mae_improvement, 2), "%")
print("RMSE Improvement:", round(rmse_improvement, 2), "%")
# VERIFY TRAIN-TEST SPLIT

print("\n========== TRAIN TEST VALIDATION ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("Training start date:", clean_data["dteday"].iloc[0])

print("Training last date:", clean_data.loc[train_mask, "dteday"].max())

print("Testing start date:", clean_data.loc[test_mask, "dteday"].min())

print("Training and testing dates overlap:",
      not set(clean_data.loc[train_mask, "dteday"]).isdisjoint(
          set(clean_data.loc[test_mask, "dteday"])
      ))
# ============================================================
# IDSE PROJECT: BIKE SHARING DEMAND ANALYSIS AND PREDICTION
# FINAL COMPLETE PIPELINE - STEP 26 ONWARDS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 26: LOAD AND PREPARE ORIGINAL MESSY DATASET
# ============================================================

print("\n========== STEP 26: FINAL DATA PREPARATION ==========")

final_data = pd.read_csv("data set/hour_messy.csv")

print("Original dataset shape:", final_data.shape)

# Remove duplicate records
final_data = final_data.drop_duplicates().copy()

# Convert date column
final_data["dteday"] = pd.to_datetime(
    final_data["dteday"], format="mixed", errors="coerce"
)

# Remove records with invalid dates
final_data = final_data.dropna(subset=["dteday"]).copy()

# Replace invalid normalized values with NaN
normalized_columns = ["temp", "atemp", "hum", "windspeed"]

for col in normalized_columns:
    final_data.loc[
        (final_data[col] < 0) | (final_data[col] > 1), col
    ] = np.nan

# Correct target count based on the assumption that
# casual and registered values are intact
final_data["cnt"] = final_data["casual"] + final_data["registered"]

# Sort data chronologically
final_data = final_data.sort_values(
    by=["dteday", "hr"]
).reset_index(drop=True)

print("Cleaned dataset shape:", final_data.shape)
print("Missing values before model imputation:",
      final_data.isnull().sum().sum())
print("Duplicate rows:", final_data.duplicated().sum())


# ============================================================
# STEP 26A: FEATURE ENGINEERING
# ============================================================

print("\n========== FEATURE ENGINEERING ==========")

final_data["hour_sin"] = np.sin(
    2 * np.pi * final_data["hr"] / 24
)

final_data["hour_cos"] = np.cos(
    2 * np.pi * final_data["hr"] / 24
)

final_data["month_sin"] = np.sin(
    2 * np.pi * (final_data["mnth"] - 1) / 12
)

final_data["month_cos"] = np.cos(
    2 * np.pi * (final_data["mnth"] - 1) / 12
)

print("Feature engineering completed.")
print("Final dataset shape:", final_data.shape)


# ============================================================
# STEP 26B: INPUT AND TARGET SELECTION
# ============================================================

print("\n========== INPUT AND TARGET SELECTION ==========")

X_final = final_data.drop(
    columns=["cnt", "casual", "registered", "instant", "dteday"]
)

y_final = final_data["cnt"]

print("Input features:", X_final.columns.tolist())
print("Input shape:", X_final.shape)
print("Target shape:", y_final.shape)


# ============================================================
# STEP 26C: CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

print("\n========== CHRONOLOGICAL TRAIN-TEST SPLIT ==========")

unique_dates = np.sort(final_data["dteday"].unique())

split_position = int(len(unique_dates) * 0.80)

train_dates = unique_dates[:split_position]
test_dates = unique_dates[split_position:]

train_mask = final_data["dteday"].isin(train_dates)
test_mask = final_data["dteday"].isin(test_dates)

X_train_final = X_final.loc[train_mask].copy()
X_test_final = X_final.loc[test_mask].copy()

y_train_final = y_final.loc[train_mask].copy()
y_test_final = y_final.loc[test_mask].copy()

print("Training records:", len(X_train_final))
print("Testing records:", len(X_test_final))

print("Training date range:",
      final_data.loc[train_mask, "dteday"].min(),
      "to",
      final_data.loc[train_mask, "dteday"].max())

print("Testing date range:",
      final_data.loc[test_mask, "dteday"].min(),
      "to",
      final_data.loc[test_mask, "dteday"].max())

print("Train-test date overlap:",
      bool(set(train_dates) & set(test_dates)))


# ============================================================
# STEP 26D: PREPROCESSING PIPELINE
# ============================================================

print("\n========== PREPROCESSING PIPELINE ==========")

numeric_features = [
    col for col in X_train_final.columns
    if col != "weathersit"
]

categorical_features = ["weathersit"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("weather", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# STEP 26E: FINAL RANDOM FOREST MODEL
# ============================================================

print("\n========== FINAL MODEL TRAINING ==========")

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

final_model.fit(X_train_final, y_train_final)

print("Final Random Forest model trained successfully!")


# ============================================================
# STEP 26F: MODEL EVALUATION
# ============================================================

print("\n========== FINAL MODEL EVALUATION ==========")

final_predictions = final_model.predict(X_test_final)

final_mae = mean_absolute_error(
    y_test_final, final_predictions
)

final_mse = mean_squared_error(
    y_test_final, final_predictions
)

final_rmse = np.sqrt(final_mse)

final_r2 = r2_score(
    y_test_final, final_predictions
)

print("Final Random Forest Results:")
print("MAE :", final_mae)
print("MSE :", final_mse)
print("RMSE:", final_rmse)
print("R2  :", final_r2)


# ============================================================
# STEP 26G: BASELINE MODEL COMPARISON
# ============================================================

print("\n========== BASELINE MODEL COMPARISON ==========")

baseline_model = DummyRegressor(strategy="mean")

baseline_model.fit(X_train_final, y_train_final)

baseline_predictions = baseline_model.predict(X_test_final)

baseline_mae = mean_absolute_error(
    y_test_final, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_test_final, baseline_predictions)
)

baseline_r2 = r2_score(
    y_test_final, baseline_predictions
)

print("Baseline MAE :", baseline_mae)
print("Baseline RMSE:", baseline_rmse)
print("Baseline R2  :", baseline_r2)

mae_improvement = (
    (baseline_mae - final_mae) / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - final_rmse) / baseline_rmse
) * 100

print("MAE Improvement (%):", mae_improvement)
print("RMSE Improvement (%):", rmse_improvement)


# ============================================================
# STEP 26H: SAVE FINAL MODEL
# ============================================================

print("\n========== SAVING FINAL MODEL ==========")

joblib.dump(final_model, "bike_sharing_final_model.pkl")

print("Final model saved successfully!")


# ============================================================
# STEP 27: FINAL MODEL VISUALIZATIONS
# ============================================================

print("\n========== STEP 27: MODEL VISUALIZATIONS ==========")

final_residuals = y_test_final.values - final_predictions

# 1. Actual vs Predicted
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_final, final_predictions, alpha=0.4
)

plt.plot(
    [y_test_final.min(), y_test_final.max()],
    [y_test_final.min(), y_test_final.max()],
    "r--"
)

plt.xlabel("Actual Bike Rentals")
plt.ylabel("Predicted Bike Rentals")
plt.title("Actual vs Predicted Bike Rentals")
plt.tight_layout()
plt.savefig("final_actual_vs_predicted.png")
plt.show()
plt.close()


# 2. Residual Plot
plt.figure(figsize=(8, 6))

plt.scatter(
    final_predictions, final_residuals, alpha=0.4
)

plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted Bike Rentals")
plt.ylabel("Residuals")
plt.title("Final Model Residual Plot")
plt.tight_layout()
plt.savefig("final_residual_plot.png")
plt.show()
plt.close()


# 3. Residual Distribution
plt.figure(figsize=(8, 6))

sns.histplot(
    final_residuals, bins=40, kde=True
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Final Model Residual Distribution")
plt.tight_layout()
plt.savefig("final_residual_distribution.png")
plt.show()
plt.close()

print("All final model graphs saved successfully!")


# ============================================================
# STEP 28: FINAL FEATURE IMPORTANCE
# ============================================================

print("\n========== STEP 28: FEATURE IMPORTANCE ==========")

rf_final = final_model.named_steps["regressor"]

preprocessor_final = final_model.named_steps["preprocessor"]

feature_names = preprocessor_final.get_feature_names_out()

importance_values = rf_final.feature_importances_

final_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
}).sort_values(
    by="Importance", ascending=False
)

print("\nFinal Feature Importance Ranking:")
print(final_importance_df.to_string(index=False))

plt.figure(figsize=(10, 7))

sns.barplot(
    data=final_importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Final Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("final_feature_importance.png")
plt.show()
plt.close()

final_importance_df.to_csv(
    "final_feature_importance.csv", index=False
)

print("Feature importance saved successfully!")


# ============================================================
# STEP 29: SAVE FINAL PREDICTION RESULTS
# ============================================================

print("\n========== STEP 29: FINAL PREDICTION RESULTS ==========")

final_prediction_results = pd.DataFrame({
    "Date": final_data.loc[test_mask, "dteday"].values,
    "Hour": final_data.loc[test_mask, "hr"].values,
    "Actual Rentals": y_test_final.values,
    "Predicted Rentals": np.round(final_predictions).astype(int)
})

final_prediction_results["Prediction Error"] = abs(
    final_prediction_results["Actual Rentals"]
    - final_prediction_results["Predicted Rentals"]
)

final_prediction_results.to_csv(
    "final_random_forest_predictions.csv",
    index=False
)

print("\nFirst 10 prediction results:")
print(final_prediction_results.head(10).to_string(index=False))

print("\nTotal predictions saved:",
      len(final_prediction_results))

print("Final prediction CSV saved successfully!")


# ============================================================
# STEP 30: LOAD AND TEST SAVED FINAL MODEL
# ============================================================

print("\n========== STEP 30: RELOAD SAVED MODEL ==========")

loaded_final_model = joblib.load(
    "bike_sharing_final_model.pkl"
)

print("Saved model loaded successfully!")

check_predictions = loaded_final_model.predict(
    X_test_final.iloc[:5]
)

print("\nFirst 5 predictions using loaded model:")

for i in range(5):
    print(
        "Actual:", y_test_final.iloc[i],
        "| Predicted:", round(check_predictions[i], 2)
    )


# ============================================================
# STEP 31: SAMPLE CUSTOM PREDICTION
# ============================================================

print("\n========== STEP 31: SAMPLE CUSTOM PREDICTION ==========")

# Use one test record as a sample input.
# You can later replace these values with your own inputs.

custom_input = X_test_final.iloc[[0]].copy()

custom_prediction = loaded_final_model.predict(custom_input)

print("Sample input details:")
print(custom_input.to_string(index=False))

print(
    "\nPredicted bike rentals for this sample:",
    round(custom_prediction[0])
)

print("\nActual bike rentals for this sample:",
      y_test_final.iloc[0])


# ============================================================
# STEP 32: FINAL PROJECT SUMMARY
# ============================================================

print("\n========== FINAL PROJECT SUMMARY ==========")

print("Project: Bike Sharing Demand Analysis and Prediction")

print("Final Model: Random Forest Regressor")

print("Training Records:", len(X_train_final))
print("Testing Records:", len(X_test_final))

print("\nFinal Performance:")
print("MAE :", round(final_mae, 2))
print("RMSE:", round(final_rmse, 2))
print("R2  :", round(final_r2, 4))

print("\nBaseline Performance:")
print("MAE :", round(baseline_mae, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R2  :", round(baseline_r2, 4))

print("\nFiles generated:")
print("1. bike_sharing_final_model.pkl")
print("2. final_random_forest_predictions.csv")
print("3. final_feature_importance.csv")
print("4. final_actual_vs_predicted.png")
print("5. final_residual_plot.png")
print("6. final_residual_distribution.png")
print("7. final_feature_importance.png")

print("\n========== PROJECT PIPELINE COMPLETED SUCCESSFULLY ==========")
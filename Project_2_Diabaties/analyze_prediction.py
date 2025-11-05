import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("./diabetes.csv")

# Check if our specific data point exists in the dataset
input_data = (0,118,84,47,230,45.8,0.551,31)
target_values = data.iloc[:, :-1].values.tolist()

# Find if this exact combination exists
matching_rows = []
for i, row in data.iloc[:, :-1].iterrows():
    if list(row.values) == list(input_data):
        matching_rows.append(i)

print(f"Found matching data point at row indices: {matching_rows}")
print(f"Original outcome: {data.iloc[matching_rows[0], -1] if matching_rows else 'Not found'}")

# Now let's see the data distribution for similar cases
print("\n=== Analysis of Similar Cases ===")

# Check cases with similar glucose levels (110-130)
similar_glucose = data[(data['Glucose'] >= 110) & (data['Glucose'] <= 130)]
print(f"Cases with glucose 110-130: {len(similar_glucose)}")
print(f"Diabetic cases in this range: {similar_glucose['Outcome'].sum()}")
print(f"Non-diabetic cases in this range: {len(similar_glucose) - similar_glucose['Outcome'].sum()}")

# Check cases with 0 pregnancies
zero_pregnancies = data[data['Pregnancies'] == 0]
print(f"\nCases with 0 pregnancies: {len(zero_pregnancies)}")
print(f"Diabetic cases with 0 pregnancies: {zero_pregnancies['Outcome'].sum()}")
print(f"Non-diabetic cases with 0 pregnancies: {len(zero_pregnancies) - zero_pregnancies['Outcome'].sum()}")

# Check cases with high BMI like our case (40-50)
high_bmi = data[(data['BMI'] >= 40) & (data['BMI'] <= 50)]
print(f"\nCases with BMI 40-50: {len(high_bmi)}")
print(f"Diabetic cases in this range: {high_bmi['Outcome'].sum()}")
print(f"Non-diabetic cases in this range: {len(high_bmi) - high_bmi['Outcome'].sum()}")

# Check cases with high insulin like our case (200-250)
high_insulin = data[(data['Insulin'] >= 200) & (data['Insulin'] <= 250)]
print(f"\nCases with insulin 200-250: {len(high_insulin)}")
print(f"Diabetic cases in this range: {high_insulin['Outcome'].sum()}")
print(f"Non-diabetic cases in this range: {len(high_insulin) - high_insulin['Outcome'].sum()}")

# Show our specific case
print("\n=== Our Specific Case ===")
print(f"Row {matching_rows[0] if matching_rows else 'Not found'}:")
print(data.iloc[matching_rows[0] if matching_rows else 0])

# Test our specific case with different models (with feature names to avoid warning)
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_df)

models = {
    'SVM': svm.SVC()
}

print("\n=== Predictions for our specific case (with feature names) ===")
for name, model in models.items():
    model.fit(data.iloc[:, :-1], data.iloc[:, -1])
    prediction = model.predict(input_data_scaled)
    print(f"{name}: {prediction[0]} ({'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'})")

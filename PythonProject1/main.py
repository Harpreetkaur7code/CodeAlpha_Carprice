import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("car data.csv")

# Remove unnecessary column
df.drop("Car_Name", axis=1, inplace=True)

# Feature Engineering
df["Current_Year"] = 2025
df["No_Years"] = df["Current_Year"] - df["Year"]
df.drop(["Year", "Current_Year"], axis=1, inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    max_depth=15
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
score = r2_score(y_test, predictions)
print(f"R2 Score: {score:.4f}")

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
import pandas as pd
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load feature columns
    model_columns = list(X.columns)

# New car data
car = {
    "Year": 2018,
    "Present_Price": 8.5,
    "Kms_Driven": 25000,
    "Owner": 0,
    "Fuel_Type_Diesel": 0,
    "Fuel_Type_Petrol": 1,
    "Seller_Type_Individual": 1,
    "Transmission_Manual": 1
}

# Convert to DataFrame
input_df = pd.DataFrame([car])

# Ensure column order matches training data
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
predicted_price = model.predict(input_df)

print(f"Predicted Car Price: ₹ {predicted_price[0]:.2f} Lakhs")
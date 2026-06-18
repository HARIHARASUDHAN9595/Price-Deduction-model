import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load Data
df = pd.read_csv("laptop_data.csv")
# Drop unwanted column
df.drop("Unnamed: 0", axis=1, inplace=True)
df = df.drop("Weight", axis=1)

# Data Clean

df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype(int)

X = df.drop("Price", axis=1)
y = df["Price"]

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Preprocessing

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer([
    ("cat", categorical_transformer, categorical_cols),
    ("num", numerical_transformer, numerical_cols)
])

# Model


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train

pipeline.fit(X_train, y_train)


# Accuracy

y_pred = pipeline.predict(X_test)

print("R2 Score:", round(r2_score(y_test, y_pred), 4))

joblib.dump(pipeline, "model.pkl")
print("Model Saved Successfully")

# User Input

print("\nEnter Laptop Details\n")

company = input("Company: ")
typename = input("TypeName: ")
inches = float(input("Inches: "))
screen = input("ScreenResolution: ")
ram = int(input("Ram (GB): "))
memory = input("Memory: ")
gpu = input("Gpu: ")

# Apple special case
if company.lower() == "apple":
    cpu = "macOS"
    opsys = "macOS"
else:
    cpu = input("Cpu: ")
    opsys = input("OpSys: ")

user_data = pd.DataFrame({
    "Company": [company],
    "TypeName": [typename],
    "Inches": [inches],
    "ScreenResolution": [screen],
    "Cpu": [cpu],
    "Ram": [ram],
    "Memory": [memory],
    "Gpu": [gpu],
    "OpSys": [opsys]
})

prediction = pipeline.predict(user_data)

print("\nPredicted Laptop Price: ₹", round(prediction[0], 2))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load dataset
df = pd.read_csv("data/winequality.csv", sep=";")

# Features & target
X = df.drop("quality", axis=1)
y = df["quality"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained and saved successfully!")
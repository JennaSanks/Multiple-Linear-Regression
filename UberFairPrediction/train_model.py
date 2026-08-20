import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("uber.csv")

print("Original dataset shape:", df.shape)


# ============================================================
# 2. REMOVE UNNECESSARY COLUMNS
# ============================================================

df = df.drop(columns=["Unnamed: 0", "key"], errors="ignore")


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

df = df.dropna()

print("After removing missing values:", df.shape)


# ============================================================
# 4. REMOVE INVALID VALUES
# ============================================================

# Passenger count should be greater than 0
df = df[df["passenger_count"] > 0]

# Fare should be positive
df = df[df["fare_amount"] > 0]


# ============================================================
# 5. CONVERT DATETIME
# ============================================================

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])


# ============================================================
# 6. EXTRACT DATE/TIME FEATURES
# ============================================================

df["hour"] = df["pickup_datetime"].dt.hour

df["day"] = df["pickup_datetime"].dt.day

df["month"] = df["pickup_datetime"].dt.month

df["year"] = df["pickup_datetime"].dt.year

df["weekday"] = df["pickup_datetime"].dt.weekday


# ============================================================
# 7. CALCULATE HAVERSINE DISTANCE
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):

    R = 6371.0  # Earth's radius in kilometers

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


df["distance"] = haversine_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)


# ============================================================
# 8. REMOVE INVALID DISTANCES
# ============================================================

df = df[df["distance"] > 0]


# ============================================================
# 9. REMOVE EXTREME / INVALID VALUES
# ============================================================

# NYC-area coordinates
df = df[
    (df["pickup_latitude"].between(40, 42)) &
    (df["dropoff_latitude"].between(40, 42)) &
    (df["pickup_longitude"].between(-75, -72)) &
    (df["dropoff_longitude"].between(-75, -72))
]

# Keep reasonable fare range
df = df[
    (df["fare_amount"] >= 2) &
    (df["fare_amount"] <= 100)
]

# Keep reasonable passenger count
df = df[
    (df["passenger_count"] <= 6)
]


print("After preprocessing:", df.shape)


# ============================================================
# 10. SELECT FEATURES
# ============================================================

features = [
    "passenger_count",
    "distance",
    "hour",
    "day",
    "month",
    "year",
    "weekday"
]

X = df[features]

y = df["fare_amount"]


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 12. CREATE MULTIPLE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 13. TRAIN MODEL
# ============================================================

print("\nTraining Multiple Linear Regression model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 14. PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\n===================================")
print("       MODEL PERFORMANCE")
print("===================================")

print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")
print(f"R2 Score : {r2:.4f}")

print("===================================")


# ============================================================
# 16. DISPLAY REGRESSION COEFFICIENTS
# ============================================================

print("\nRegression Coefficients:")

for feature, coefficient in zip(
    features,
    model.coef_
):
    print(
        f"{feature:20} : {coefficient:.6f}"
    )

print("\nIntercept:", model.intercept_)


# ============================================================
# 17. SAVE MODEL
# ============================================================

pickle.dump(
    model,
    open("UberFareModel.pkl", "wb")
)

print("\nModel saved successfully!")
print("File: UberFareModel.pkl")
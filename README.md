# Data-science-projects


# ----------------------------------------------
# ENERGY CONSUMPTION PREDICTION PROJECT 
# ----------------------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset

data = {
    'Temperature': [15, 18, 21, 24, 27, 30, 33, 36, 39, 42],
    'Humidity': [30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
    'Light': [200, 250, 300, 350, 400, 450, 500, 550, 600, 650],
    'Appliances': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Energy_Consumption': [100, 120, 135, 160, 180, 200, 230, 250, 280, 310]
}

df = pd.DataFrame(data)
print("\n Dataset Loaded Successfully!\n")
print(df.head())

# Step 3: Data Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df)
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x='Temperature', y='Energy_Consumption', data=df, color='blue')
plt.title("Energy Consumption vs Temperature")
plt.show()

# Step 4: Prepare Data
X = df[['Temperature', 'Humidity', 'Light', 'Appliances']]
y = df['Energy_Consumption']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n Data Split into Train and Test sets")

# Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n Model Trained Successfully!")

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate Model
print("\n Model Evaluation Results:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 8: Predict New Data
new_data = pd.DataFrame({
    'Temperature': [28],
    'Humidity': [55],
    'Light': [420],
    'Appliances': [6]
})
pred = model.predict(new_data)
print("\n Predicted Energy Consumption:", round(pred[0], 2))

# Step 9: Visualization - Actual vs Predicted
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual Energy Consumption")
plt.ylabel("Predicted Energy Consumption")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()

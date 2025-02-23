# house
house price prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset (replace with actual dataset path)
data = pd.DataFrame({
    'area': [1500, 1800, 2400, 3000, 3500, 4000, 4200],
    'bedrooms': [3, 4, 3, 5, 4, 5, 6],
    'bathrooms': [2, 3, 2, 3, 3, 4, 5],
    'price': [300000, 360000, 450000, 540000, 600000, 720000, 750000]
})

# Define features and target variable
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Predict price for a new house
def predict_price(area, bedrooms, bathrooms):
    return model.predict(np.array([[area, bedrooms, bathrooms]]))[0]

# Example prediction
new_price = predict_price(3200, 4, 3)
print(f"Predicted Price for a house with 3200 sqft, 4 bedrooms, 3 bathrooms: ${new_price:.2f}")

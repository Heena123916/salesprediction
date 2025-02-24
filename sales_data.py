# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'sales_data.csv' with your file)
# The dataset should have columns like 'advertising_budget', 'season', 'store_traffic', and 'sales'
df = pd.read_csv('sales_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Select features (independent variables) and target (dependent variable)
# For example, you might want to predict 'sales' based on features like 'advertising_budget', 'season', 'store_traffic'
X = df[['advertising_budget', 'season', 'store_traffic']]  # Independent variables
y = df['sales']  # Dependent variable (sales)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

# Plot actual vs predicted sales
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()

# Predict sales for a new set of input features (example)
new_data = np.array([[5000, 2, 100]])  # Example: advertising_budget=5000, season=2, store_traffic=100
new_sales_prediction = model.predict(new_data)
print(f'Predicted Sales for new data: {new_sales_prediction[0]}')

# Task 1 - Sales & Demand Forecasting
# Future Interns - FUTURE_ML_01

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('train.csv', encoding='latin1')
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Data cleaning
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Month'] = df['Order Date'].dt.to_period('M')

# Monthly sales
monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()
monthly_sales['Month'] = monthly_sales['Month'].astype(str)
print("\nMonthly Sales:")
print(monthly_sales.head(10))

# Sales Trend Graph
plt.figure(figsize=(14,6))
plt.plot(monthly_sales['Month'], monthly_sales['Sales'],
         marker='o', color='blue', linewidth=2)
plt.title('Monthly Sales Trend (2015-2018)', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('sales_trend.png')
plt.show()

# Forecasting Model
monthly_sales['Index'] = range(len(monthly_sales))
X = monthly_sales[['Index']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X, y)

# Predict next 6 months
future_index = np.array([[len(monthly_sales)+i] for i in range(6)])
predictions = model.predict(future_index)

print("\nNext 6 months Sales Forecast:")
for i, pred in enumerate(predictions):
    print(f"Month {i+1}: ${pred:,.2f}")

# Forecast Graph
plt.figure(figsize=(14,6))
plt.plot(monthly_sales['Month'], monthly_sales['Sales'],
         marker='o', color='blue', linewidth=2, label='Actual Sales')
future_months = [f'Future Month {i+1}' for i in range(6)]
plt.plot(future_months, predictions,
         marker='s', color='red', linewidth=2,
         linestyle='--', label='Forecasted Sales')
plt.title('Sales Forecast - Next 6 Months', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('sales_forecast.png')
plt.show()
print("Task 1 Complete!")

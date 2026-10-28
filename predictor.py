import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# ==========================================
# PHASE 1: DATA LOADING & CLEANING PIPELINE
# ==========================================
raw_file = 'household_power_consumption.txt'
clean_csv = 'electricity_data.csv'

print("Loading raw text data...")
if not os.path.exists(raw_file):
    raise FileNotFoundError(f"Missing file: Please place '{raw_file}' in this folder.")

# Read the semicolon-separated file safely
df = pd.read_csv(raw_file, sep=';', low_memory=False)

print("Cleaning missing values and converting types...")
# Replace dataset missing markers ('?') with None and drop them
df = df.replace('?', None).dropna()

# Convert the consumption column from text to numeric values safely
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'])

# Convert strings directly to date formats before grouping to prevent parsing drift
df['Parsed_Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

print("Aggregating minute data into daily Kilowatt-hours...")
# The raw data is power per minute. Summing it and multiplying by (1/60) converts it to true kWh.
daily_df = df.groupby('Parsed_Date')['Global_active_power'].sum().reset_index()
daily_df.columns = ['Date', 'Total_Power_Minute_Sum']
daily_df['Total_Power'] = daily_df['Total_Power_Minute_Sum'] / 60.0
daily_df = daily_df.drop(columns=['Total_Power_Minute_Sum'])

# Save to CSV as required by the assignment guidelines
daily_df.to_csv(clean_csv, index=False)
print(f"Data pipeline complete. Cleaned file saved as '{clean_csv}'.")


# ==========================================
# PHASE 2: FEATURE ENGINEERING
# ==========================================
print("\nEngineering time-series features...")
daily_df['Date'] = pd.to_datetime(daily_df['Date'])

# Create Lag Features (What did they consume yesterday and 7 days ago?)
daily_df['Power_Yesterday'] = daily_df['Total_Power'].shift(1)
daily_df['Power_Last_Week'] = daily_df['Total_Power'].shift(7)

# Create Calendar Features
daily_df['Day_of_Week'] = daily_df['Date'].dt.dayofweek.astype(str)

# Clean up the empty rows created by time shifts
daily_df = daily_df.dropna().reset_index(drop=True)


# ==========================================
# PHASE 3: CHRONOLOGICAL DATA SPLIT
# ==========================================
print("Performing chronological train-test split...")
X_raw = daily_df[['Power_Yesterday', 'Power_Last_Week', 'Day_of_Week']]
y = daily_df['Total_Power']

# Calculate the index for a perfect 80/20 chronological split
split_idx = int(len(daily_df) * 0.8)

# Convert categorical days safely using pandas dummy indicators (dtype=int ensures 0 and 1 outputs)
X_encoded = pd.get_dummies(X_raw, columns=['Day_of_Week'], drop_first=True, dtype=int)

X_train, X_test = X_encoded.iloc[:split_idx], X_encoded.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# ==========================================
# PHASE 4: MODEL TRAINING & EVALUATION
# ==========================================
print("Training Linear Regression vs Decision Tree...")

# Model A: Linear Regression (Required)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)

# Model B: Decision Tree Regressor (Optional comparison for extra credit)
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_predictions)

print("\n PERFORMANCE METRICS:")
print(f" Linear Regression MAE : {lr_mae:.3f} kWh per day")
print(f" Decision Tree MAE     : {dt_mae:.3f} kWh per day")


# ==========================================
# PHASE 5: FUTURE BILL ESTIMATION
# ==========================================
print("\nGenerating sample future bill estimates (Next 30 Test Days)...")
# Assume a realistic utility tariff rate per kWh ($0.15)
UTILITY_RATE = 0.15 

# Take the first 30 days of our test period as our simulated 'future month'
future_lr_usage = lr_predictions[:30].sum()
future_dt_usage = dt_predictions[:30].sum()

print(f"Estimated Bill (Linear Regression) : ${future_lr_usage * UTILITY_RATE:.2f}")
print(f"Estimated Bill (Decision Tree)     : ${future_dt_usage * UTILITY_RATE:.2f}")
    

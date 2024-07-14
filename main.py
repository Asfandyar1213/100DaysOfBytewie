import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import scipy.stats as stats

# Load datasets
plant_1_gen = pd.read_csv('Plant_1_Generation_Data.csv')
plant_1_weather = pd.read_csv('Plant_1_Weather_Sensor_Data.csv')
plant_2_gen = pd.read_csv('Plant_2_Generation_Data.csv')
plant_2_weather = pd.read_csv('Plant_2_Weather_Sensor_Data.csv')

# Convert DATE_TIME to datetime
plant_1_gen['DATE_TIME'] = pd.to_datetime(plant_1_gen['DATE_TIME'])
plant_1_weather['DATE_TIME'] = pd.to_datetime(plant_1_weather['DATE_TIME'])
plant_2_gen['DATE_TIME'] = pd.to_datetime(plant_2_gen['DATE_TIME'])
plant_2_weather['DATE_TIME'] = pd.to_datetime(plant_2_weather['DATE_TIME'])

# Merge data
plant_1 = pd.merge(plant_1_gen, plant_1_weather, on='DATE_TIME', how='inner')
plant_2 = pd.merge(plant_2_gen, plant_2_weather, on='DATE_TIME', how='inner')

# Verify merging
print(plant_1.head())
print(plant_1.info())
print(plant_2.head())
print(plant_2.info())

# Feature engineering
plant_1['HOUR'] = plant_1['DATE_TIME'].dt.hour
plant_2['HOUR'] = plant_2['DATE_TIME'].dt.hour

# Select features and target
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'HOUR']
target = 'DC_POWER'

X_plant_1 = plant_1[features]
y_plant_1 = plant_1[target]

X_plant_2 = plant_2[features]
y_plant_2 = plant_2[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_plant_1, y_plant_1, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Plot predictions vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual DC Power')
plt.ylabel('Predicted DC Power')
plt.title('Actual vs Predicted DC Power')
plt.show()

# Residuals plot
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title('Residuals Distribution')
plt.show()

# Correlation heatmap
corr_matrix = plant_1.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Time series plot of power output
plt.plot(plant_1['DATE_TIME'], plant_1['DC_POWER'])
plt.xlabel('Date Time')
plt.ylabel('DC Power')
plt.title('Time Series of DC Power Output')
plt.show()

# Probability distribution of power output
sns.histplot(plant_1['DC_POWER'], kde=True)
plt.title('Probability Distribution of DC Power Output')
plt.show()

# Confidence interval for mean power output
mean_dc_power = np.mean(plant_1['DC_POWER'])
std_dev_dc_power = np.std(plant_1['DC_POWER'])
confidence_interval = stats.norm.interval(0.95, loc=mean_dc_power, scale=std_dev_dc_power/np.sqrt(len(plant_1['DC_POWER'])))

print(f'95% Confidence Interval for Mean DC Power Output: {confidence_interval}')

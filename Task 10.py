import pandas as pd

# Load dataset
df = pd.read_csv("titanic.csv")

# Check for missing values
print(df.isnull().sum())

# Choose imputation strategy (e.g., mean imputation)
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Alternative strategies:
# Median imputation
# df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Mode imputation (for categorical variables)
# df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Dropping rows/columns (consider data loss implications)
# df.dropna(subset=["Embarked"], inplace=True)  # Drop rows with missing "Embarked"
# df.drop("Cabin", axis=1, inplace=True)  # Drop "Cabin" column with many missing values



import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load dataset
df = pd.read_csv("car_evaluation.csv")

# One-hot encoding
categorical_columns = df.select_dtypes(include=["object"]).columns
ohe = OneHotEncoder(sparse=False)
encoded_df = pd.concat([df.drop(categorical_columns, axis=1),
                        pd.DataFrame(ohe.fit_transform(df[categorical_columns]))], axis=1)

# Label encoding (consider information loss)
le = LabelEncoder()
df["buy"] = le.fit_transform(df["buy"])
df["maint"] = le.fit_transform(df["maint"])
df["doors"] = le.fit_transform(df["doors"])

# Compare results: Analyze the encoded data to see if one-hot encoding provides
# more information for the model.




import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_csv("wine_quality.csv")

# Feature scaling (normalization)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = pd.DataFrame(scaler.fit_transform(df))
scaled_df.columns = df.columns

# Standardization
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df))
scaled_df.columns = df.columns

# Analyze distribution: Use visualizations (histograms) to compare the
# distributions of the original and scaled features.




import pandas as pd
from scipy import stats

# Load dataset
df = pd.read_csv("boston_housing.csv")

# Z-score
z_scores = stats.zscore(df)
outliers = df[(z_scores > 3.5) | (z_scores < -3.5)].any(axis=1)
df_filtered = df[~outliers]

# IQR (Interquartile Range)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
df_filtered = df[~outliers.any(axis=1)]

# Visualization methods (boxplots, scatter plots)
import matplotlib.pyplot as plt

plt.boxplot(df["price"])  # Identify potential outliers visually

# Handle outliers based on domain knowledge and analysis results.





import pandas as pd
from sklearn.impute import KNNImputer, MICE  # For KNN and MICE imputation

# Load dataset
df = pd.read_csv("retail_sales.csv")

# KNN imputation
imputer = KNNImputer(n_neighbors=5)  # Experiment with different n_neighbors
df_imputed = pd.DataFrame(imputer.fit_transform(df))
df_imputed.columns = df.columns

# MICE (Multiple Imputation by Chained Equations)
imputer = MICE(n_imput





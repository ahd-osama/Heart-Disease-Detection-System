import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("/content/drive/My Drive/Heart_Disease_Detection/data/raw_data.csv")

# Handling missing values
print("Before handling missing values:")
print(data.isnull().sum())

data['oldpeak'] = data['oldpeak'].fillna(data['oldpeak'].median())
data['restecg'] = data['restecg'].fillna(data['restecg'].mode()[0])

print("\nAfter handling missing values:")
print(data.isnull().sum())

# Normalize numerical features
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

print("Before Normalization:")
print(data[numerical_cols])

scaler = MinMaxScaler()

data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

print("\nAfter Normalization:")
print(data[numerical_cols])

# Feature selction
corr_matrix = data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

correlation = data.corr()['target'].drop('target').abs()

threshold = 0.15

selected_features = correlation[correlation >= threshold].index.tolist()

print("Selected Features:", selected_features)

target_data = data['target']

data = data[selected_features]

# Encode categorical features
categorical_cols = ['cp', 'thal', 'slope']

data = pd.get_dummies(data, columns=categorical_cols)

data[data.select_dtypes('bool').columns] = data.select_dtypes('bool').astype(int)

data['target'] = target_data

print("\nColumns after encoding:")
print(data.columns)

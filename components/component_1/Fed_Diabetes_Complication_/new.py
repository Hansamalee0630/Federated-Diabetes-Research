import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your processed file
df = pd.read_csv("datasets/complication_dataset/processed_data.csv")

# 1. Box Plots (Demographics & Outliers)
# This shows the spread across Gender and highlights the outlier patterns
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x='GENDER', y='AGE', hue='GENDER', data=df, palette='Set2', legend=False)
plt.title('Age Spread by Gender')

plt.subplot(1, 2, 2)
sns.boxplot(x='GENDER', y='CR', hue='GENDER', data=df, palette='Set3', legend=False)
plt.title('Creatinine (CR) Outlier Patterns by Gender')
plt.savefig('boxplots.png')

# 2. Correlation Heatmap (Low Linear Correlations)
plt.figure(figsize=(10, 8))
# Selecting only numeric columns from your dataset
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap: Low Linear Correlations between Features')
plt.savefig('heatmap.png')

# 3. Demographic Distribution (Age/Gender)
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='AGE', hue='GENDER', kde=True, multiple="stack")
plt.title('Age and Gender Distribution')
plt.savefig('distribution.png')

print("Visuals saved: boxplots.png, heatmap.png, distribution.png")
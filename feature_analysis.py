import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Reading data
csv_file = 'Variables_Horno.csv'
dl = pandas.read_csv(csv_file)
dl.head()
d0 = dl.values

ncol = len(dl.columns)
y = d0[:, 0]
x = d0[:, 1:ncol-20]
x2 = d0[:, 1:ncol]

# Calculate correlation matrix for feature selection
print("Calculating feature correlations with target variable...")
corr_matrix = pandas.DataFrame(d0).corr()

# Get correlation with target variable (first column)
target_correlations = corr_matrix[0].iloc[1:]

# Sort and display top correlations
print("\nFeature Correlations with Target Variable:")
sorted_correlations = target_correlations.abs().sort_values(ascending=False)
print(sorted_correlations.head(10))  # Show top 10 correlations

# Visualize correlations with target
plt.figure(figsize=(12, 8))
target_correlations.plot(kind='bar')
plt.title('Feature Correlation with Target Variable')
plt.xlabel('Feature Index')
plt.ylabel('Correlation Coefficient')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.tight_layout()

# Plot a heatmap for the most correlated features
plt.figure(figsize=(10, 8))
# Select top 10 correlated features plus the target
top_features = [0] + sorted_correlations.nlargest(10).index.tolist()
sns.heatmap(corr_matrix.iloc[top_features, top_features],
            annot=True,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            fmt='.2f')
plt.title('Correlation Matrix of Target and Top 10 Features')
plt.tight_layout()

# Create subplots with proper layout
fig, axs = plt.subplots(10, 1, figsize=(10, 15), sharex=True)
axs[0].plot(y)
axs[0].set_title('Target Variable')
axs[0].set_ylabel('Value')

# Plot the rest of subplots in a loop
for i in range(1, 10):
    feature_idx = ncol - (21 - i + 1)  # Calculate correct feature index
    axs[i].plot(x2[:, feature_idx])
    axs[i].set_ylabel(f'Feature {feature_idx}')

    # Add x-label only to the bottom subplot
    if i == 9:
        axs[i].set_xlabel('Sample Index')

fig.suptitle('Key Variables Visualization', fontsize=16)
plt.tight_layout()

plt.show()
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_percentage_error,
                             mean_absolute_error,
                             r2_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reading data
csv_file = 'Variables_Horno.csv'
dl = pandas.read_csv(csv_file)
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

# Select features with highest absolute correlation but keep original sign info
print("\nTop Feature Correlations with Target Variable (with direction):")
sorted_correlations = target_correlations.abs().sort_values(ascending=False)
print(target_correlations.loc[sorted_correlations.nlargest(10).index])

# Define polynomial combinations to try
poly_combinations = {
    'x': [1],
    'x²': [2],
    'x³': [3],
    'x + x²': [1, 2],
    'x + x³': [1, 3],
    'x² + x³': [2, 3],
    'x + x² + x³': [1, 2, 3]
}

# Store results for different feature counts
all_results = {}
best_overall = {'r2': -1, 'feature_count': 0, 'combo': '', 'predictions': None}

# Iterate over different feature counts
for feature_count in [2, 3, 4]:
    print(f"\n\n{'='*50}")
    print(f"TRAINING MODELS WITH TOP {feature_count} FEATURES")
    print(f"{'='*50}")

    # Select top N features
    selected_features = sorted_correlations.nlargest(feature_count).index.tolist()
    X = dl.iloc[:, selected_features].values

    print(f"Selected features: {selected_features}")
    nan_counts = np.isnan(X).sum(axis=0)
    print(f"NaN counts per feature: {nan_counts}")

    # Split data - use same random_state for consistent comparisons
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Store results for this feature count
    results = {}
    best_score = -1
    best_combo = None

    # Try different polynomial combinations
    for combo_name, degree_list in poly_combinations.items():

        # Create polynomial features based on combination
        X_poly_train = np.empty((X_train.shape[0], 0))
        X_poly_test = np.empty((X_test.shape[0], 0))

        for degree in degree_list:
            if degree == 1:
                X_poly_train = np.hstack((X_poly_train, X_train))
                X_poly_test = np.hstack((X_poly_test, X_test))
            else:
                X_poly_train = np.hstack((X_poly_train, X_train ** degree))
                X_poly_test = np.hstack((X_poly_test, X_test ** degree))

        # Create and train pipeline
        poly_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=5000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                learning_rate_init=0.001,
                random_state=42
            ))
        ])

        poly_pipeline.fit(X_poly_train, y_train)
        y_pred = poly_pipeline.predict(X_poly_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Save results
        poly_feature_count = X_poly_train.shape[1]
        results[combo_name] = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'predictions': y_pred,
            'poly_feature_count': poly_feature_count
        }

        print(f"  {combo_name} - R²: {r2:.4f}, MSE: {mse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2%}")

        # Update best model for this feature count
        if r2 > best_score:
            best_score = r2
            best_combo = combo_name

        # Update best overall model
        if r2 > best_overall['r2']:
            best_overall['r2'] = r2
            best_overall['feature_count'] = feature_count
            best_overall['combo'] = combo_name
            best_overall['predictions'] = y_pred

    # Store all results for this feature count
    all_results[feature_count] = {
        'results': results,
        'best_combo': best_combo,
        'best_score': best_score,
        'y_test': y_test  # Store y_test for plotting
    }

    print(f"\nBest model for {feature_count} features: {best_combo} (R² = {best_score:.4f})")

# Print overall best model
print(f"\n\n{'='*50}")
print(f"BEST OVERALL MODEL: {best_overall['feature_count']} features with {best_overall['combo']}")
print(f"R² = {best_overall['r2']:.4f}")
print(f"{'='*50}")

# Create subplot grid for each feature count
for feature_count, data in all_results.items():
    # Create a 3x3 grid (with 7 used)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Models with Top {feature_count} Features", fontsize=16)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    # Plot each polynomial combination in its own subplot
    for i, (combo_name, result) in enumerate(data['results'].items()):
        ax = axes[i]
        y_test = data['y_test']
        y_pred = result['predictions']
        r2 = result['r2']

        # Plot scatter of predicted vs actual
        ax.scatter(y_test, y_pred, alpha=0.6)

        # Plot perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5)

        # Highlight this as best if applicable
        if combo_name == data['best_combo']:
            ax.set_title(f"{combo_name} (R² = {r2:.4f})\n★ BEST ★", fontweight='bold')
            rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                fill=False, edgecolor='red', linestyle='-', linewidth=2)
            ax.add_patch(rect)
        else:
            ax.set_title(f"{combo_name} (R² = {r2:.4f})")

        # Add labels and grid
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.grid(True, linestyle='--', alpha=0.7)

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    plt.show()

# Compare R² values across all combinations and feature counts
plt.figure(figsize=(14, 8))
x_pos = np.arange(len(poly_combinations))
bar_width = 0.25
opacity = 0.8
colors = plt.cm.tab10.colors

# Plot R² for each feature count
for i, feature_count in enumerate([2, 3, 4]):
    r2_values = [all_results[feature_count]['results'][combo]['r2'] for combo in poly_combinations]
    plt.bar(
        x_pos + i*bar_width,
        r2_values,
        bar_width,
        alpha=opacity,
        color=colors[i],
        label=f'Top {feature_count} Features'
    )

# Add labels and styling
plt.xlabel('Polynomial Combination')
plt.ylabel('R² Score')
plt.title('R² Comparison by Feature Count and Polynomial Combination')
plt.xticks(x_pos + bar_width, poly_combinations.keys(), rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Compare best models from each feature count
fig, ax = plt.subplots(figsize=(10, 10))

# Plot perfect prediction line
y_test = all_results[best_overall['feature_count']]['y_test']
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='Perfect Prediction')

# Plot best model from each feature count
for i, feature_count in enumerate([2, 3, 4]):
    best_combo = all_results[feature_count]['best_combo']
    best_preds = all_results[feature_count]['results'][best_combo]['predictions']
    best_r2 = all_results[feature_count]['best_score']
    y_test = all_results[feature_count]['y_test']

    ax.scatter(
        y_test,
        best_preds,
        alpha=0.7,
        color=colors[i],
        s=60,
        label=f"Top {feature_count}: {best_combo} (R² = {best_r2:.4f})"
    )

# Highlight overall best model
ax.scatter(
    all_results[best_overall['feature_count']]['y_test'],
    best_overall['predictions'],
    s=100,
    facecolors='none',
    edgecolors='black',
    linewidth=2,
    label=f"Best Overall: {best_overall['feature_count']} features, {best_overall['combo']}"
)

# Style plot
ax.set_title("Comparison of Best Models for Each Feature Count")
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
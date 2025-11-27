import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
import os

output_folder = './output/correlation/'
os.makedirs(output_folder, exist_ok=True)

numerical_features = [
    'GaugeDist_m', 
    'StormDur_H', 'StormAccum_mm', 'StormAvgI_mm/h', 
    'Peak_I15_mm/h', 'Peak_I30_mm/h', 'Peak_I60_mm/h',
    'ContributingArea_km2', 
    'PropHM23', 'dNBR/1000', 'KF', 'Acc015_mm', 
    'Acc030_mm', 'Acc060_mm'
]

from data import PWFDF_Data
data = PWFDF_Data()
df = data.df
print(df['Fire_ID'])
quit()

# Calculate point-biserial correlation for each feature with binary Response
correlations = {}
p_values = {}

for feature in numerical_features:
    # Remove any NaN values for this feature
    valid_data = df[[feature, 'Response']].dropna()
    
    # Calculate point-biserial correlation
    corr, p_val = pointbiserialr(valid_data['Response'], valid_data[feature])
    correlations[feature] = corr
    p_values[feature] = p_val

# Create a DataFrame for better visualization
corr_df = pd.DataFrame({
    'Feature': list(correlations.keys()),
    'Correlation': list(correlations.values()),
    'P-value': list(p_values.values())
})
corr_df = corr_df.sort_values('Correlation', ascending=False, key=abs)

# Print correlation table
print("=" * 80)
print("CORRELATION WITH RESPONSE (Point-Biserial Correlation)")
print("=" * 80)
print(f"{'Feature':<25} {'Correlation':>12} {'P-value':>12} {'Significance':>15}")
print("-" * 80)

for _, row in corr_df.iterrows():
    significance = '***' if row['P-value'] < 0.001 else ('**' if row['P-value'] < 0.01 else ('*' if row['P-value'] < 0.05 else 'ns'))
    print(f"{row['Feature']:<25} {row['Correlation']:>12.4f} {row['P-value']:>12.4e} {significance:>15}")

print("-" * 80)
print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
print("=" * 80)

# Visualization 1: Bar plot of correlations
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Bar plot
colors = ['red' if c < 0 else 'green' for c in corr_df['Correlation']]
axes[0].barh(corr_df['Feature'], corr_df['Correlation'], color=colors, alpha=0.7)
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
axes[0].set_xlabel('Point-Biserial Correlation', fontsize=12)
axes[0].set_title('Correlation between Features and Response', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# Add correlation values on bars
for idx, row in corr_df.iterrows():
    x_pos = row['Correlation'] + (0.02 if row['Correlation'] > 0 else -0.02)
    ha = 'left' if row['Correlation'] > 0 else 'right'
    axes[0].text(x_pos, row['Feature'], f"{row['Correlation']:.3f}", 
                va='center', ha=ha, fontsize=9)

# Visualization 2: Heatmap style
corr_matrix = corr_df.set_index('Feature')[['Correlation']]
sns.heatmap(corr_matrix.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
            cbar_kws={'label': 'Correlation'}, ax=axes[1], 
            vmin=-1, vmax=1, linewidths=0.5)
axes[1].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(output_folder + 'correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: Create boxplots to visualize distribution differences
n_features = len(numerical_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for idx, feature in enumerate(numerical_features):
    df_clean = df[[feature, 'Response']].dropna()
    
    # Create boxplot
    response_0 = df_clean[df_clean['Response'] == 0][feature]
    response_1 = df_clean[df_clean['Response'] == 1][feature]
    
    axes[idx].boxplot([response_0, response_1], labels=['Response=0', 'Response=1'])
    axes[idx].set_title(f"{feature}\n(r={correlations[feature]:.3f})", fontsize=10)
    axes[idx].set_ylabel(feature)
    axes[idx].grid(axis='y', alpha=0.3)

# Remove empty subplots
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(output_folder + 'feature_distributions_by_response.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nVisualizations saved as:")
print("- correlation_analysis.png")
print("- feature_distributions_by_response.png")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('ts_rmsd.csv')

# Extract RMSD column
rmsd = df['RMSD(A)']

# Remove any NaN values
rmsd_clean = rmsd[~rmsd.isna()]

# Calculate statistics
mean_rmsd = rmsd_clean.mean()
median_rmsd = rmsd_clean.median()
std_rmsd = rmsd_clean.std()

print(f"RMSD Statistics:")
print(f"  N = {len(rmsd_clean)}")
print(f"  Mean = {mean_rmsd:.4f} A")
print(f"  Median = {median_rmsd:.4f} A")
print(f"  Std Dev = {std_rmsd:.4f} A")
print(f"  Min = {rmsd_clean.min():.4f} A")
print(f"  Max = {rmsd_clean.max():.4f} A")

# Create density histogram
plt.figure(figsize=(6, 5))
plt.hist(rmsd_clean, bins=40, density=True, alpha=0.7, color='lightblue')

# Add vertical lines for mean and median
plt.axvline(mean_rmsd, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_rmsd:.3f} A')
plt.axvline(median_rmsd, color='orange', linestyle='--', linewidth=2, label=f'Median = {median_rmsd:.3f} A')

plt.xlim(0, rmsd_clean.max() * 1.1)
plt.xlabel('RMSD (A)', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=12)
#plt.title('Distribution of RMSD between GFN2-xTB and B3LYP-D3 Transition State Geometries', fontsize=12)
plt.legend(fontsize=12)
#plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save as PDF and PNG
plt.savefig('ts_rmsd_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('ts_rmsd_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n RMSD distribution plot saved as ts_rmsd_distribution.pdf and .png")

plt.close()

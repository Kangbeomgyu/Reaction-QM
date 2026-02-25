import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Read the CSV file
df = pd.read_csv('common_reaction_info.csv')

# Extract relevant columns
gfn_de_dagger = df['dE_dagger(GFN)']
b3lyp_de_dagger = df['dE_dagger(B3LYP-D3)']
gfn_dh = df['dH(GFN)']
b3lyp_dh = df['dH(B3LYP-D3)']

# Remove any NaN values
mask_de = ~(gfn_de_dagger.isna() | b3lyp_de_dagger.isna())

gfn_de_dagger_clean = gfn_de_dagger[mask_de]
b3lyp_de_dagger_clean = b3lyp_de_dagger[mask_de]

mask_dh = ~(gfn_dh.isna() | b3lyp_dh.isna())
gfn_dh_clean = gfn_dh[mask_dh]
b3lyp_dh_clean = b3lyp_dh[mask_dh]

low_indices_gfn = np.where(gfn_de_dagger_clean<0.5)[0].tolist()
low_indices_dft = np.where(b3lyp_de_dagger_clean<0.5)[0].tolist()

print (len(low_indices_gfn))
print (len(low_indices_dft))

print (low_indices_gfn[0])
print (low_indices_dft[0])

print (gfn_de_dagger_clean[low_indices_gfn[0]])

# Calculate correlation coefficients
r_de, p_de = stats.pearsonr(b3lyp_de_dagger_clean, gfn_de_dagger_clean)
mae_de = np.mean(np.abs(gfn_de_dagger_clean - b3lyp_de_dagger_clean))
rmse_de = np.sqrt(np.mean((gfn_de_dagger_clean - b3lyp_de_dagger_clean)**2))

r_dh, p_dh = stats.pearsonr(b3lyp_dh_clean, gfn_dh_clean)
mae_dh = np.mean(np.abs(gfn_dh_clean - b3lyp_dh_clean))
rmse_dh = np.sqrt(np.mean((gfn_dh_clean - b3lyp_dh_clean)**2))

low_indices_gfn = np.where(np.abs(gfn_dh_clean)<0.5)[0].tolist()
low_indices_dft = np.where(np.abs(b3lyp_dh_clean)<0.5)[0].tolist()

print (len(low_indices_gfn))
print (len(low_indices_dft))


# Get axis limits
de_min_val = min(b3lyp_de_dagger_clean.min(), gfn_de_dagger_clean.min())
de_max_val = max(b3lyp_de_dagger_clean.max(), gfn_de_dagger_clean.max())
dh_min_val = min(b3lyp_dh_clean.min(), gfn_dh_clean.min())
dh_max_val = max(b3lyp_dh_clean.max(), gfn_dh_clean.max())

print (de_min_val, de_max_val, dh_min_val, dh_max_val)

print("=" * 60)
print("PLOT 1: Activation Energy (dE_dagger)")
print("=" * 60)

# === PLOT 1A: Scatter points only (PNG) ===
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(b3lyp_de_dagger_clean, gfn_de_dagger_clean, 
           s=0.5, alpha=0.5, c='skyblue', edgecolors='none')
ax.set_xlim(0, de_max_val)
ax.set_ylim(0, de_max_val)
ax.axis('off')  # Remove all axes
fig.patch.set_alpha(0)  # Transparent background
ax.patch.set_alpha(0)
plt.tight_layout(pad=0)
plt.savefig('activation_energy_points.png', dpi=300, bbox_inches='tight', 
            transparent=True, pad_inches=0)
print("? Scatter points saved: activation_energy_points.png")
plt.close()

# === PLOT 1B: Axes, labels, legend only (PDF) ===
fig, ax = plt.subplots(figsize=(5, 5))
# Draw diagonal line
ax.plot([0, de_max_val], [0, de_max_val], 'r--', linewidth=1, alpha=0.5)
ax.set_xlim(0, de_max_val)
ax.set_ylim(0, de_max_val)
ax.set_xlabel('B3LYP-D3 (kcal/mol)', fontsize=14)
ax.set_ylabel('GFN2-xTB (kcal/mol)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend([f'$R^2$ = {r_de**2:.4f}\nMAE = {mae_de:.2f} kcal/mol\nRMSE = {rmse_de:.2f} kcal/mol'], 
          loc='upper left', fontsize=12)
#ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('activation_energy_frame.pdf', bbox_inches='tight')
print("? Frame (axes/labels) saved: activation_energy_frame.pdf")
print(f"  N = {len(b3lyp_de_dagger_clean)}")
print(f"  R^2 = {r_de**2:.4f}")
print(f"  MAE = {mae_de:.2f} kcal/mol")
print(f"  RMSE = {rmse_de:.2f} kcal/mol")
plt.close()

print("\n" + "=" * 60)
print("PLOT 2: Reaction Enthalpy (dH)")
print("=" * 60)

# === PLOT 2A: Scatter points only (PNG) ===
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(b3lyp_dh_clean, gfn_dh_clean, 
           s=0.5, alpha=0.5, c='skyblue', edgecolors='none')
ax.set_xlim(dh_min_val, dh_max_val)
ax.set_ylim(dh_min_val, dh_max_val)
ax.axis('off')  # Remove all axes
fig.patch.set_alpha(0)  # Transparent background
ax.patch.set_alpha(0)
plt.tight_layout(pad=0)
plt.savefig('reaction_enthalpy_points.png', dpi=300, bbox_inches='tight', 
            transparent=True, pad_inches=0)
print("? Scatter points saved: reaction_enthalpy_points.png")
plt.close()

# === PLOT 2B: Axes, labels, legend only (PDF) ===
fig, ax = plt.subplots(figsize=(5, 5))
# Draw diagonal line
ax.plot([dh_min_val, dh_max_val], [dh_min_val, dh_max_val], 'r--', linewidth=1, alpha=0.5)
ax.set_xlim(dh_min_val, dh_max_val)
ax.set_ylim(dh_min_val, dh_max_val)
ax.set_xlabel('B3LYP-D3 (kcal/mol)', fontsize=14)
ax.set_ylabel('GFN2-xTB (kcal/mol)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend([f'$R^2$ = {r_dh**2:.4f}\nMAE = {mae_dh:.2f} kcal/mol\nRMSE = {rmse_dh:.2f} kcal/mol'], 
          loc='upper left', fontsize=12)
#ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reaction_enthalpy_frame.pdf', bbox_inches='tight')
print("? Frame (axes/labels) saved: reaction_enthalpy_frame.pdf")
print(f"  N = {len(b3lyp_dh_clean)}")
print(f"  R^2 = {r_dh**2:.4f}")
print(f"  MAE = {mae_dh:.2f} kcal/mol")
print(f"  RMSE = {rmse_dh:.2f} kcal/mol")
plt.close()


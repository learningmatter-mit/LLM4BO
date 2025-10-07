from Bio.Align import substitution_matrices
blosum62_matrix = substitution_matrices.load("BLOSUM62")
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys
protein_seq = sys.argv[1]
import numpy as np
import pandas as pd
import os
df = pd.read_csv(f'data/{protein_seq}/fitness.csv')
df['fitness'] = (df['fitness'] - df['fitness'].min())/(df['fitness'].max() - df['fitness'].min())
subdf = df.sort_values(by='fitness', ascending=False).head(50)
mean_blosum = lambda x,y : np.mean([blosum62_matrix[(x[i],y[i])] for i in range(4)]) if x != y else 0
n_top = len(subdf)
blosum_matrix = np.full((n_top, n_top), np.nan)  # Fill with NaN

for i, seq1 in enumerate(subdf['Combo']):
    for j, seq2 in enumerate(subdf['Combo']):
        if i >= j:  # Only fill bottom triangle (including diagonal)
            blosum_matrix[i, j] = mean_blosum(seq1, seq2)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Main histogram with log scale
ax.hist(df['fitness'], bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Normalized Fitness', fontsize=12)
ax.set_ylabel('Frequency (log scale)', fontsize=12)
ax.set_yscale('log')
ax.set_title('Distribution of Fitness Values with BLOSUM62 Heatmap (Top 50)', fontsize=14)
ax.grid(True, alpha=0.3)

# Create inset for heatmap in top right
inset_ax = inset_axes(ax, width="35%", height="35%", loc='upper right', 
                     bbox_to_anchor=(-0.05, -0.1, 1, 1), bbox_transform=ax.transAxes)

# Create heatmap with specified range [-2, 4], masking upper triangle
masked_matrix = np.ma.masked_invalid(blosum_matrix)  # Mask NaN values (upper triangle)
im = inset_ax.imshow(masked_matrix, cmap='RdYlBu_r', vmin=-2, vmax=4, aspect='auto')

# Customize inset
inset_ax.set_title('BLOSUM62 Mean\n(Top 50 Sequences)', fontsize=10, pad=10)
inset_ax.set_xlabel('Sequence Index', fontsize=8)
inset_ax.set_ylabel('Sequence Index', fontsize=8)
inset_ax.tick_params(axis='both', which='major', labelsize=7)

# Add smaller colorbar to inset
cbar_ax = inset_axes(inset_ax, width="4%", height="60%", loc='center right', 
                     bbox_to_anchor=(-0.1, 0.07, 1, 1), bbox_transform=inset_ax.transAxes)
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.set_label('BLOSUM62 Score', fontsize=7)
cbar.ax.tick_params(labelsize=6)

# Add some statistics as text
stats_text = f"Total sequences: {len(df)}\n"
stats_text += f"Top 50 mean fitness: {subdf['fitness'].mean():.3f}\n"
stats_text += f"Median fitness: {df['fitness'].median():.4f}"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
os.makedirs('distributions/raw', exist_ok=True)
plt.savefig(f'distributions/raw/{protein_seq}.png')
plt.close()
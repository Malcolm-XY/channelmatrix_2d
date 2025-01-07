# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 18:44:40 2025

@author: 18307
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_mapping_with_values_and_significance(resolutions, manual, ortho, stereo, feature_name, significance):
    bar_width = 0.2
    x = np.arange(len(resolutions))

    # Plot each mapping type
    manual_bars = plt.bar(x - bar_width, manual, width=bar_width, label='Manual Mapping', color='blue')
    ortho_bars = plt.bar(x, [val if val is not None else 0 for val in ortho], 
                          width=bar_width, label='Orthographic', color='orange')
    stereo_bars = plt.bar(x + bar_width, [val if val is not None else 0 for val in stereo], 
                           width=bar_width, label='Stereographic', color='green')

    # Display values on bars
    def display_values(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only display if value is greater than 0
                plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{height:.2f}', 
                         ha='center', va='bottom', fontsize=9)

    display_values(manual_bars)
    display_values(ortho_bars)
    display_values(stereo_bars)

    # Add comparison line
    comparison_value = manual[0]  # Manual mapping accuracy at resolution=9
    plt.axhline(y=comparison_value, color='red', linestyle='--', label=f'Manual Mapping (Res.=9)')

    # Add significance annotations
    for i, sigs in enumerate(significance):
        if sigs is not None:
            for j, sig in enumerate(sigs):
                if sig:
                    # Define comparison group pairs
                    if j == 1:  # Manual vs. Orthographic
                        x1, x2 = x[i] - bar_width, x[i]
                        offset = 1  # Base height offset for manual-ortho
                    elif j == 2:  # Manual vs. Stereographic
                        x1, x2 = x[i] - bar_width, x[i] + bar_width
                        offset = 3  # Slightly higher for manual-stereo
                    else:
                        continue

                    # Adjust y_pos dynamically based on bar heights and offset
                    y_pos = max([manual[i], (ortho[i] if ortho[i] is not None else 0), 
                                 (stereo[i] if stereo[i] is not None else 0)]) + 2 + offset

                    # Add a line and significance marker
                    plt.plot([x1, x1, x2, x2], [y_pos - 0.5, y_pos, y_pos, y_pos - 0.5], color='black')
                    plt.text((x1 + x2) / 2, y_pos, sig, ha='center', va='bottom', color='black', fontsize=10)

    # Add labels, title, and legend
    plt.xlabel('Resolution')
    plt.ylabel('Average Accuracy (%)')
    plt.title(f'Mapping Method Comparison Using {feature_name} Feature')
    plt.xticks(x, resolutions)
    plt.ylim(70, 105)  # Adjust y-axis range for better visibility
    
    # Move legend to the left
    plt.legend(loc='upper left') #, bbox_to_anchor=(-0.3, 0.5), frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

# %% Data
# Data for DE_LDS
de_resolutions = ['9x9', '16x16', '24x24']
de_manual = [85.4233, 86.4384, 85.2844]
de_ortho = [None, 89.2955, 89.6837]
de_stereo = [None, 88.9812, 89.4098]

# Data for PSD_LDS
psd_resolutions = ['9x9', '16x16', '24x24']
psd_manual = [83.0121, 81.2357, 79.6903]
psd_ortho = [None, 83.5426, 82.3593]
psd_stereo = [None, 84.2085, 82.1128]

# Significance data for DE_LDS and PSD_LDS
de_significance = [[None, None, None], [None, '**', '**'], [None, '**', '**']]
psd_significance = [[None, None, None], [None, None, None], [None, None, None]]

# Test with DE_LDS
plot_mapping_with_values_and_significance(
    de_resolutions, de_manual, de_ortho, de_stereo, "DE_LDS", de_significance
)

# Test with PSD_LDS
plot_mapping_with_values_and_significance(
    psd_resolutions, psd_manual, psd_ortho, psd_stereo, "PSD_LDS", psd_significance
)

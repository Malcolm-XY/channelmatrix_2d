# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:14:36 2025

@author: 18307
"""

import matplotlib.pyplot as plt
import numpy as np

# Function to plot data
def plot_mapping_comparison(resolutions, manual, ortho, stereo, feature_name):
    bar_width = 0.2
    x = np.arange(len(resolutions))

    # Plot each mapping type
    plt.bar(x - bar_width, manual, width=bar_width, label='Manual Mapping', color='blue')
    plt.bar(x, [val if val is not None else 0 for val in ortho], 
            width=bar_width, label='Orthographic', color='orange')
    plt.bar(x + bar_width, [val if val is not None else 0 for val in stereo], 
            width=bar_width, label='Stereographic', color='green')

    # Add labels, title, and legend
    plt.xlabel('Resolution')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{feature_name} Mapping Type Comparison')
    plt.xticks(x, resolutions)
    plt.ylim(70, 100)  # Adjust y-axis range for better visibility
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Function to plot data with a comparison line
def plot_mapping_with_comparison_line(resolutions, manual, ortho, stereo, feature_name):
    bar_width = 0.2
    x = np.arange(len(resolutions))

    # Plot each mapping type
    plt.bar(x - bar_width, manual, width=bar_width, label='Manual Mapping', color='blue')
    plt.bar(x, [val if val is not None else 0 for val in ortho], 
            width=bar_width, label='Orthographic', color='orange')
    plt.bar(x + bar_width, [val if val is not None else 0 for val in stereo], 
            width=bar_width, label='Stereographic', color='green')

    # Add comparison line
    comparison_value = manual[0]  # Manual mapping accuracy at resolution=9
    plt.axhline(y=comparison_value, color='red', linestyle='--', label=f'Manual Mapping (Resolution=9): {comparison_value:.2f}')

    # Add labels, title, and legend
    plt.xlabel('Resolution')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{feature_name} Mapping Type Comparison')
    plt.xticks(x, resolutions)
    plt.ylim(70, 100)  # Adjust y-axis range for better visibility
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Function to plot data with comparison line and significance annotations
def plot_mapping_with_significance(resolutions, manual, ortho, stereo, feature_name, significance):
    bar_width = 0.2
    x = np.arange(len(resolutions))

    # Plot each mapping type
    plt.bar(x - bar_width, manual, width=bar_width, label='Manual Mapping', color='blue')
    plt.bar(x, [val if val is not None else 0 for val in ortho], 
            width=bar_width, label='Orthographic', color='orange')
    plt.bar(x + bar_width, [val if val is not None else 0 for val in stereo], 
            width=bar_width, label='Stereographic', color='green')

    # Add comparison line
    comparison_value = manual[0]  # Manual mapping accuracy at resolution=9
    plt.axhline(y=comparison_value, color='red', linestyle='--', label=f'Manual Mapping (Res.=9x9): {comparison_value:.2f}')

    # Add significance annotations
    for i, sigs in enumerate(significance):
        if sigs is not None:
            for j, sig in enumerate(sigs):
                if sig:
                    x_pos = x[i] + (j - 1) * bar_width  # Adjust position for ortho (-1) and stereo (+1)
                    plt.text(x_pos, comparison_value + 1, sig, ha='center', va='bottom', color='black', fontsize=10)

    # Add labels, title, and legend
    plt.xlabel('Resolution')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{feature_name} Mapping Type Comparison')
    plt.xticks(x, resolutions)
    plt.ylim(70, 100)  # Adjust y-axis range for better visibility
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Function to plot data with significance and display values on bars
def plot_mapping_with_values_and_significance(resolutions, manual, ortho, stereo, feature_name, significance):
    bar_width = 0.2
    x = np.arange(len(resolutions))

    # Plot each mapping type
    manual_bars = plt.bar(x - bar_width, manual, width=bar_width, label='Manual Mapping', color='blue')
    ortho_bars = plt.bar(x, [val if val is not None else 0 for val in ortho], 
                          width=bar_width, label='Orthographic', color='orange')
    stereo_bars = plt.bar(x + bar_width, [val if val is not None else 0 for val in stereo], 
                           width=bar_width, label='Stereographic', color='green')

    # Add comparison line
    comparison_value = manual[0]  # Manual mapping accuracy at resolution=9
    plt.axhline(y=comparison_value, color='red', linestyle='--') #, label=f'Manual Mapping (Res.={resolutions[0]})')

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

    # Add significance annotations
    for i, sigs in enumerate(significance):
        if sigs is not None:
            for j, sig in enumerate(sigs):
                if sig:
                    x_pos = x[i] + (j - 1) * bar_width  # Adjust position for ortho (-1) and stereo (+1)
                    y_pos = max([manual[i], (ortho[i] if ortho[i] is not None else 0), (stereo[i] if stereo[i] is not None else 0)]) + 2
                    plt.text(x_pos, y_pos, sig, ha='center', va='bottom', color='black', fontsize=10)

    # Add labels, title, and legend
    plt.xlabel('Resolution')
    plt.ylabel('Average Accuracy (%)')
    plt.title(f'Mapping Method Comparison Using {feature_name} Feature')
    plt.xticks(x, resolutions)
    plt.ylim(70, 100)  # Adjust y-axis range for better visibility
    plt.legend()
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

# %% Plot
# plot_mapping_comparison(de_resolutions, de_manual, de_ortho, de_stereo, "DE_LDS")
# plot_mapping_comparison(psd_resolutions, psd_manual, psd_ortho, psd_stereo, "PSD_LDS")

# %% Plot with acomparison line
# plot_mapping_with_comparison_line(de_resolutions, de_manual, de_ortho, de_stereo, "DE_LDS")
# plot_mapping_with_comparison_line(psd_resolutions, psd_manual, psd_ortho, psd_stereo, "PSD_LDS")

# %%  Plot with significance annotations
# plot_mapping_with_significance(de_resolutions, de_manual, de_ortho, de_stereo, "DE_LDS", de_significance)
# plot_mapping_with_significance(psd_resolutions, psd_manual, psd_ortho, psd_stereo, "PSD_LDS", psd_significance)

# %%  Plot with values and significance annotations
plot_mapping_with_values_and_significance(de_resolutions, de_manual, de_ortho, de_stereo, "DE_LDS", de_significance)
plot_mapping_with_values_and_significance(psd_resolutions, psd_manual, psd_ortho, psd_stereo, "PSD_LDS", psd_significance)

import matplotlib.patches as patches

def add_ttest_arrows(ax, x1, x2, y, p_value_label):
    """
    Add arrows and significance labels to indicate t-test comparisons.
    """
    ax.annotate('', xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle='-[', color='black', lw=1.5))
    ax.text((x1 + x2) / 2, y + 0.5, p_value_label, ha='center', va='bottom', fontsize=10, color='black')

# 示例位置：手动添加两组数据的对比
x1_pos = 0.8  # 起始柱形 x 坐标
x2_pos = 1.2  # 结束柱形 x 坐标
comparison_y = 90  # 箭头 y 坐标
p_value = "**"

fig, ax = plt.subplots()
# 在图中调用
add_ttest_arrows(ax, x1_pos, x2_pos, comparison_y, p_value)

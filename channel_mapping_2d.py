# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 01:45:40 2024

@author: 18307
"""

import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import griddata

def draw_projection(projection, sample_index=0):
    plt.imshow(projection[sample_index], cmap='viridis')  # 选择颜色映射 'viridis'
    plt.colorbar()  # 添加颜色条
    plt.title("Matrix Visualization using imshow")
    plt.show()  

def mapping_2d(
        data, distribution, resolution, rounding_method='floor', 
        interpolation=False, interp_method='linear', imshow=False
        ):
    """
    Maps data to a 2D grid matrix of size resolution^2 based on distribution coordinates.
    """
    # Deep copy inputs to ensure safety
    distribution = copy.deepcopy(distribution)
    data = np.copy(data)
    
    # Normalize or shift distribution to ensure all coordinates are within [0, 1]
    x_min, y_min = np.min(distribution['x']), np.min(distribution['y'])
    x_max, y_max = np.max(distribution['x']), np.max(distribution['y'])
    
    x_shift = -x_min if x_min < 0 else 0
    y_shift = -y_min if y_min < 0 else 0
    distribution['x'] = np.array(distribution['x']) + x_shift
    distribution['y'] = np.array(distribution['y']) + y_shift
    distribution['x'] /= (x_max + x_shift)
    distribution['y'] /= (y_max + y_shift)

    # Map the data
    len_samples, size_data = data.shape[0], data.shape[1]
    
    grid_temp_global = np.zeros((len_samples, resolution, resolution))
    for i in range(len_samples):
        grid_temp_local = np.zeros((resolution, resolution))
        for j in range(size_data):
            x = distribution['x'][j] * (resolution - 1)
            y = distribution['y'][j] * (resolution - 1)
            x, y = _apply_rounding(x, y, rounding_method)
            if 0 <= x < resolution and 0 <= y < resolution:
                grid_temp_local[x][y] = data[i][j]
        grid_temp_global[i] = grid_temp_local

# # %%
#     mapped_points = set()
#     overlap_counts = defaultdict(int)
    
#     for j in range(len(distribution['x'])):
#         x = distribution['x'][j] * (resolution - 1)
#         y = distribution['y'][j] * (resolution - 1)
#         x, y = _apply_rounding(x, y, rounding_method)

#         if 0 <= x < resolution and 0 <= y < resolution:
#             if (x, y) in mapped_points:
#                 overlap_counts[(x, y)] += 1
#             else:
#                 mapped_points.add((x, y))
    
#     print("Mapping Results:")
#     print(f"- Total grid points: {resolution * resolution}")
#     print(f"- Mapped grid points: {len(mapped_points)}")
#     print(f"- Overlap occurrences: {len(overlap_counts)}")
#     for point, count in overlap_counts.items():
#         print(f"  Overlap at grid point {point}: {count + 1} times")

    # interpolation
    if interpolation:
        grid_temp_global = fill_zeros_with_interpolation(grid_temp_global, resolution, interp_method)
    
    # plt   
    if imshow: draw_projection(grid_temp_global)
    
    return grid_temp_global

def orthographic_projection_2d(
        data, distribution, resolution, rounding_method='floor', 
        interpolation=False, interp_method='linear', imshow=False
        ):
    """
    Map data to a 2D grid based on distribution coordinates and apply orthographic projection.
    """
    # Deep copy inputs to ensure safety
    distribution = copy.deepcopy(distribution)
    data = np.copy(data)

    # Normalize to ensure all corrdinates are witin [0, 1]
    x_min, y_min = np.min(distribution['x']), np.min(distribution['y'])
    x_max, y_max = np.max(distribution['x']), np.max(distribution['y'])
    distribution['x'] = (np.array(distribution['x']) - x_min) / (x_max - x_min)
    distribution['y'] = (np.array(distribution['y']) - y_min) / (y_max - y_min)

    # Map the data
    len_samples, size_data = data.shape[0], data.shape[1]

    grid_temp_global = np.zeros((len_samples, resolution, resolution))
    for i in range(len_samples):
        grid_temp_local = np.zeros((resolution, resolution))
        for j in range(size_data):
            x = distribution['x'][j] * (resolution - 1)
            y = distribution['y'][j] * (resolution - 1)
            x, y = _apply_rounding(x, y, rounding_method)
            if 0 <= x < resolution and 0 <= y < resolution:
                grid_temp_local[x][y] += data[i][j]
        grid_temp_global[i] = grid_temp_local

    # interpolation
    if interpolation:
        grid_temp_global = fill_zeros_with_interpolation(grid_temp_global, resolution, interp_method)
    
    # plt   
    if imshow: draw_projection(grid_temp_global)
    
    return grid_temp_global

def stereographic_projection_2d(
        data, distribution, resolution, rounding_method='floor', 
        interpolation=False, interp_method='linear', prominence=0.1, epsilon=0.01, imshow=False
        ):
    """
    Perform stereographic projection from 3D points to a 2D grid with optimized computation.
    """
    # Deep copy inputs to ensure safety
    distribution = copy.deepcopy(distribution)
    data = np.copy(data)

    x_coords, y_coords, z_coords = np.array(distribution['x']), np.array(distribution['y']), np.array(distribution['z'])
    z_coords = (z_coords - np.min(z_coords)) / (np.max(z_coords) - np.min(z_coords)) - prominence

    x_proj = x_coords / (1 - z_coords + epsilon)
    y_proj = y_coords / (1 - z_coords + epsilon)

    x_norm = (x_proj - np.min(x_proj)) / (np.max(x_proj) - np.min(x_proj))
    y_norm = (y_proj - np.min(y_proj)) / (np.max(y_proj) - np.min(y_proj))

    # mapping
    grid_temp_global = np.zeros((data.shape[0], resolution, resolution))
    for i in range(data.shape[0]):
        grid_temp_local = np.zeros((resolution, resolution))
        for j in range(len(x_norm)):
            x, y = x_norm[j] * (resolution - 1), y_norm[j] * (resolution - 1)
            x, y = _apply_rounding(x, y, rounding_method)
            if 0 <= x < resolution and 0 <= y < resolution:
                grid_temp_local[x, y] += data[i, j]
        grid_temp_global[i] = grid_temp_local
    
    # interpolation
    if interpolation:
        grid_temp_global = fill_zeros_with_interpolation(grid_temp_global, resolution, interp_method)
    
    # plt   
    if imshow: draw_projection(grid_temp_global)
    
    return grid_temp_global

def _apply_rounding(x, y, method):
    if method == 'floor':
        return math.floor(x), math.floor(y)
    elif method == 'ceil':
        return math.ceil(x), math.ceil(y)
    elif method == 'round':
        return round(x), round(y)
    raise ValueError("Invalid rounding method. Choose 'floor', 'ceil', or 'round'.")

# def fill_zeros_with_interpolation(data, resolution, interp_method='linear'):
#     filled_data = np.copy(data)
#     for sample_idx in range(data.shape[0]):
#         sample = data[sample_idx]
#         non_zero_coords = np.array(np.nonzero(sample)).T
#         non_zero_values = sample[non_zero_coords[:, 0], non_zero_coords[:, 1]]
#         grid_x, grid_y = np.mgrid[0:resolution, 0:resolution]
#         grid_points = np.array([grid_x.flatten(), grid_y.flatten()]).T
#         filled_values = griddata(non_zero_coords, non_zero_values, grid_points, method=interp_method, fill_value=0)
#         filled_data[sample_idx] = filled_values.reshape(resolution, resolution)
#     return filled_data

from joblib import Parallel, delayed

def fill_zeros_with_interpolation(data, resolution, interp_method='linear', n_jobs=-1):
    """
    使用并行化的方式对多个样本进行插值填充。

    参数：
        data (ndarray): 输入数据，形状为 (num_samples, resolution, resolution)。
        resolution (int): 网格分辨率。
        interp_method (str): 插值方法，例如 'linear', 'nearest', 'cubic'。
        n_jobs (int): 并行线程数，-1 表示使用所有 CPU 核心。

    返回：
        filled_data (ndarray): 插值填充后的数据。
    """
    def interpolate_sample(sample):
        non_zero_coords = np.array(np.nonzero(sample)).T
        if len(non_zero_coords) == 0:
            return sample  # 如果样本全为零，直接返回
        non_zero_values = sample[non_zero_coords[:, 0], non_zero_coords[:, 1]]
        grid_x, grid_y = np.mgrid[0:resolution, 0:resolution]
        grid_points = np.array([grid_x.flatten(), grid_y.flatten()]).T
        filled_values = griddata(non_zero_coords, non_zero_values, grid_points, method=interp_method, fill_value=0)
        return filled_values.reshape(resolution, resolution)

    # 使用并行处理每个样本
    filled_data = Parallel(n_jobs=n_jobs)(
        delayed(interpolate_sample)(data[sample_idx]) for sample_idx in range(data.shape[0])
    )
    return np.array(filled_data)


# %% Example Usage
if __name__ == '__main__':
    import utils

    # data and distribution
    data_sample = utils.get_channel_feature_mat('de_LDS', 'gamma', 'sub1ex1')
    dist_auto = utils.get_distribution('auto')
    dist_manual = utils.get_distribution('manual')

    # manual orthograph
    sample_ortho_manual = mapping_2d(data_sample, dist_manual, 9, interpolation=False, imshow=True)
    # sample_ortho_manual_interpolated = mapping_2d(data_sample, dist_manual, 9, interpolation=True, interp_method='linear')

    # auto orthograph
    sample_2d_or = orthographic_projection_2d(data_sample, dist_auto, 16, interpolation=False, imshow=True)
    # sample_2d_or_interpolated = orthographic_projection_2d(data_sample, dist_auto, 16, interpolation=True, interp_method='linear', imshow=True)

    # auto stereographic
    sample_2d_st = stereographic_projection_2d(data_sample, dist_auto, 16, interpolation=False, prominence=0.1, imshow=True)
    # sample_2d_st_interpolated = stereographic_projection_2d(data_sample, dist_auto, 16, interpolation=True, interp_method='linear', prominence=0.1, imshow=True)
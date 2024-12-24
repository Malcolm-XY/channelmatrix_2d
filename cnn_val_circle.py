# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 01:23:16 2024

@author: 18307
"""
import os
import numpy as np
import pandas as pd

import utils
import cnn_validation

def cnn_validation_circle(model, mapping_func, dist, resolution, interp, feature, subject_range, experiment_range):
    labels = utils.get_label()
    distribution = utils.get_distribution(dist)
    
    results_entry = []
    for sub in subject_range:
        for ex in experiment_range:
            file_name = f'sub{sub}ex{ex}.mat'
            print(f'Processing {file_name}...')
            data_alpha = utils.get_channel_feature_mat(feature, 'alpha', f'sub{sub}ex{ex}')
            data_beta = utils.get_channel_feature_mat(feature, 'beta', f'sub{sub}ex{ex}')
            data_gamma = utils.get_channel_feature_mat(feature, 'gamma', f'sub{sub}ex{ex}')

            alpha_mapped = mapping_func(data_alpha, distribution, resolution, interpolation=interp)
            beta_mapped = mapping_func(data_beta, distribution, resolution, interpolation=interp)
            gamma_mapped = mapping_func(data_gamma, distribution, resolution, interpolation=interp, imshow=True)

            alpha_mapped = utils.safe_normalize(alpha_mapped)
            beta_mapped = utils.safe_normalize(beta_mapped)
            gamma_mapped = utils.safe_normalize(gamma_mapped)
            
            data_mapped = np.stack((alpha_mapped, beta_mapped, gamma_mapped), axis=1)
            
            result = cnn_validation.cnn_validation(model, data_mapped, labels)
            
            # Add identifier to the result
            result['Identifier'] = f'sub{sub}ex{ex}'
            results_entry.append(result)

    # print(f'Final Results: {results_entry}')
    print('K-Fold Validation compelete\n')
    
    return results_entry

def cnn_cross_validation_circle(model, mapping_func, dist, resolution, interp, feature, subject_range, experiment_range):
    labels = utils.get_label()
    distribution = utils.get_distribution(dist)
    
    results_entry = []
    for sub in subject_range:
        for ex in experiment_range:
            file_name = f'sub{sub}ex{ex}.mat'
            print(f'Processing {file_name}...')
            data_alpha = utils.get_channel_feature_mat(feature, 'alpha', f'sub{sub}ex{ex}')
            data_beta = utils.get_channel_feature_mat(feature, 'beta', f'sub{sub}ex{ex}')
            data_gamma = utils.get_channel_feature_mat(feature, 'gamma', f'sub{sub}ex{ex}')

            alpha_mapped = mapping_func(data_alpha, distribution, resolution, interpolation=interp)
            beta_mapped = mapping_func(data_beta, distribution, resolution, interpolation=interp)
            gamma_mapped = mapping_func(data_gamma, distribution, resolution, interpolation=interp, imshow=True)

            alpha_mapped = utils.safe_normalize(alpha_mapped)
            beta_mapped = utils.safe_normalize(beta_mapped)
            gamma_mapped = utils.safe_normalize(gamma_mapped)
            
            data_mapped = np.stack((alpha_mapped, beta_mapped, gamma_mapped), axis=1)
            
            result = cnn_validation.cnn_cross_validation(model, data_mapped, labels)
            
            # Add identifier to the result
            result['Identifier'] = f'sub{sub}ex{ex}'
            results_entry.append(result)

    # print(f'Final Results: {results_entry}')
    print('K-Fold Validation compelete\n')
    
    return results_entry

# %% Usage
# from Models import models
import channel_mapping_2d
from Models import models_multiscale

# model = models.EnhancedCNN2DModel2(channels=3)
model = models_multiscale.MultiScaleCNN()
mapping_func = channel_mapping_2d.orthographic_projection_2d
# mapping_func = channel_mapping_2d.stereographic_projection_2d
distribution, resolution, interp = 'manual', 32, False

feature, subject_range, experiment_range = 'de_LDS', range(1, 16), range(1, 4)

results = cnn_cross_validation_circle(
    model, mapping_func, distribution, resolution, interp, feature, subject_range, experiment_range)

# Save to xlsx
results_df = pd.DataFrame(results)
columns_order = ['Identifier'] + [col for col in results_df.columns if col != 'Identifier']
results_df = results_df[columns_order]

# File name
output_path = os.path.join(
    os.getcwd(),
    'Results',
    f"{mapping_func.__name__[:3]}_dist_{distribution}_res_{resolution}_interp_{interp}.xlsx"
)
results_df.to_excel(output_path, index=False, sheet_name='K-Fold Results')

# # Shutdown computer
# print("Program completed. Shutting down...")
# os.system("shutdown /s /t 1")
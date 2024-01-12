# Demo for decay analysis using the DecayFitNet Toolbox
import sys
import torch
import numpy as np
import torchaudio
import os
from pathlib import Path
import matplotlib.pyplot as plt

from toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from toolbox.utils import calc_mse
from toolbox.core import discard_last_n_percent, decay_model, PreprocessRIR

from toolbox.BayesianDecayAnalysis import BayesianDecayAnalysis

import pandas as pd

def main(path_to_dataset):
    print(f'path_to_dataset: {path_to_dataset}')
    if not os.path.isdir(path_to_dataset):
        print("Warning: path to dataset does not exist. Check if root or dataset_folder is correct")
        sys.exit()

    # Load meta data in CSV
    csv_file = os.path.join(path_to_dataset, 'metadata_archive.csv')

    # Create dataframe and add the fold index
    df = pd.read_csv(csv_file)
    df["rt60"] = df["rt60"].apply(lambda x: float(x.strip('[]')))
    
    # ===============================================================================
    # Parameters
    fs = 48000
    
    n_slopes = 0  # 0 = number of active slopes is determined by network or bayesian analysis (between 1 and 3)

    filter_frequencies = [125, 250, 500, 1000, 2000, 4000]

    # Bayesian paramters: a_range and n_range are both exponents, i.e., actual range = 10^a_range or 10^n_range
    parameter_ranges = {'t_range': [0.1, 3.5],
                        'a_range': [-3, 0],
                        'n_range': [-10, -2]}
    n_iterations = 100

    # ===============================================================================
    # Prepare the model
    decayfitnet = DecayFitNetToolbox(n_slopes=n_slopes, sample_rate=fs, filter_frequencies=filter_frequencies)
    
    # Init Bayesian decay analysis
    bda = BayesianDecayAnalysis(n_slopes, fs, parameter_ranges, n_iterations)
    
    mse_avg_DFN = []
    mse_freqband_DFN = torch.empty(len(df),len(filter_frequencies))
    
    # mse_avg_bayes = []
    # mse_freqband_bayes = torch.empty(len(df),len(filter_frequencies))

    idx = 0
    for filename in df['filename']:
        # ===============================================================================
        # Parameters
        rir_fname = os.path.basename(filename)
        # print(os.path.join(path_to_dataset, rir_fname))
        
        fadeout_length = 0  # in seconds

        # Load impulse response
        rir, fs = torchaudio.load(os.path.join(path_to_dataset, rir_fname))

        # Use only omni channel
        if len(rir.shape) > 1:
            rir = rir[0:1, :]

        # Delete potential fade-out windows
        if fadeout_length > 0:
            rir = rir[:, 0:round(-fadeout_length * fs)]

        # ===============================================================================
        # Get ground truth EDCs from raw RIRs:

        # Init Preprocessing
        rir_preprocessing = PreprocessRIR(sample_rate=fs, filter_frequencies=filter_frequencies)
        # Schroeder integration, analyse_full_rir: if RIR onset should be detected, set this to False
        true_edc, __ = rir_preprocessing.schroeder(rir, analyse_full_rir=True)
        time_axis = (torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) / fs)
        # Permute into [n_bands, n_batches, n_samples]
        true_edc = true_edc.permute(1, 0, 2)

        # ===============================================================================
        # Analyze with DecayFitNet

        # Process: analyse_full_rir: if RIR onset should be detected, set this to False
        estimated_parameters_decayfitnet, norm_vals_decayfitnet = decayfitnet.estimate_parameters(rir, analyse_full_rir=True)

        # Get fitted EDC from estimated parameters
        fitted_edc_decayfitnet = decay_model(torch.from_numpy(estimated_parameters_decayfitnet[0]),
                                             torch.from_numpy(estimated_parameters_decayfitnet[1]),
                                             torch.from_numpy(estimated_parameters_decayfitnet[2]),
                                             time_axis=time_axis,
                                             compensate_uli=True,
                                             backend='torch')

        # Discard last 5% for MSE evaluation
        true_edc = discard_last_n_percent(true_edc, 5)
        fitted_edc_decayfitnet = discard_last_n_percent(fitted_edc_decayfitnet, 5)

        # Calculate MSE between true EDC and fitted EDC
        mse_per_frequencyband = calc_mse(true_edc, fitted_edc_decayfitnet, verbose=False)
        
        mse_freqband_DFN[idx, :] = mse_per_frequencyband.view(1, -1)
        mse_avg = torch.mean(mse_per_frequencyband)
        mse_avg_DFN.append(mse_avg.item())
        
        idx += 1
        
#         # # ===============================================================================
#         # Bayesian decay analysis

#         # Estimate decay parameters
#         estimated_parameters_bayesian, norm_vals_bayesian = bda.estimate_parameters(rir)

#         # Get EDC from estimated parameters
#         fitted_edc_bayesian = decay_model(estimated_parameters_bayesian[0],
#                                           estimated_parameters_bayesian[1],
#                                           estimated_parameters_bayesian[2],
#                                           time_axis=time_axis,
#                                           compensate_uli=True,
#                                           backend='np')
#         fitted_edc_bayesian = torch.from_numpy(np.sum(fitted_edc_bayesian, 1, keepdims=True))  # sum up terms of the decay model

#         # Discard last 5% for MSE evaluation
#         fitted_edc_bayesian = discard_last_n_percent(fitted_edc_bayesian, 5)

#         # Calculate MSE between true EDC and fitted EDC
#         mse_per_frequencyband = calc_mse(true_edc, fitted_edc_bayesian)
    df['MSE_DFN Average'] = mse_avg_DFN
    df1 = pd.DataFrame(mse_freqband_DFN, columns=['MSE_DFN %u Hz' % f for f in filter_frequencies])
    
    df = pd.concat([df, df1], axis=1)
    df.to_csv(os.path.join(path_to_dataset, 'metadata_DecayFitNet.csv'), index=False)
    
if __name__ == "__main__":
    
    # data_dict = {"mit": "../data", "frl": "../data", "arni": "Z:/elec/t412-asp"} 
    audio_path = "C:/Users/jacki/Desktop/Thesis/rircomplete/data/Pyroomacoustics/ism_dar/ism_dar_1"
    
    audio_path = "/scratch/elec/t412-asp/ism_dar_1" 
    main(audio_path)
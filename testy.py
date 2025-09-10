import tensorflow as tf
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import pathlib
import seaborn as sns

from keras import layers
from keras import models
import matplotlib.pyplot as plt
from datetime import datetime

sampling_rate = 1600  #Hz
train_size = 0.8
time_period = 10
test_size = 0.1
valid_size = 0.1
random_state = 42
interpolation_interval = 0.120 #s
block_interval = 7 #s
padding_time = 1.56 #s


def find_next_file_index(label, subdir):
    """Find the next available file index"""
    try:
        existing_files = os.listdir(subdir)
        if existing_files:
            indices = []
            for file in existing_files:
                if file.startswith(label) and '_' in file:
                    try:
                        index = int(file.split('_')[-1].split('.')[0])
                        indices.append(index)
                    except ValueError:
                        continue
            
            if indices:
                return max(indices) + 1
            else:
                return 1
        else:
            return 1
    except Exception:
        return 1


def write_csv(processed_df, label, output_directory):
    """Write processed dataframes to CSV files"""
    subdir = os.path.join(output_directory, label)
    os.makedirs(subdir, exist_ok=True)
    
    next_index = find_next_file_index(label, subdir)
    
    for i, df in enumerate(processed_df):
        for j in range(5):
            filename = f"{label}_{next_index + i*5 + j}.csv"
            file_path = os.path.join(subdir, filename)
            
            label_df = pd.DataFrame([[label]], columns=[df.columns[0]])
            result_df = pd.concat([label_df, df], ignore_index=True)
            result_df.to_csv(file_path, index=False)


def process_csv_file(file_path):
    """Process a single CSV file"""
    # Read CSV file
    df = pd.read_csv(file_path, header=None, encoding='utf-8', skiprows=1, usecols=range(1, 3))
    df.columns = ['DMM-1 Time (s)', 'DMM-1 Current (A)']
    
    # Normalize time to start from 0
    df['DMM-1 Time (s)'] = df['DMM-1 Time (s)'] - df['DMM-1 Time (s)'].min()
    
    # Interpolate the function
    interpolation_func = interp1d(df['DMM-1 Time (s)'], df['DMM-1 Current (A)'], 
                                 kind='linear', fill_value='extrapolate')
    max_time = df['DMM-1 Time (s)'].max()
    min_time = df['DMM-1 Time (s)'].min()
    new_time = np.arange(min_time, max_time, interpolation_interval)
    new_current = interpolation_func(new_time)
    
    interpolated_df = pd.DataFrame({'Time (s)': new_time, 'Current (A)': new_current})
    
    # Create time blocks with padding
    num_blocks = int(np.floor(max_time / block_interval))
    time_intervals = [(i * block_interval, (i + 1) * block_interval) for i in range(num_blocks)]
    dfs_block_with_padding = []
    
    for start_time, end_time in time_intervals:
        # Get block data
        block_df = interpolated_df[
            (interpolated_df['Time (s)'] >= start_time) & 
            (interpolated_df['Time (s)'] <= end_time)
        ]
        
        # Add padding
        padding_before_time = pd.Series(np.arange(start_time - padding_time, start_time, interpolation_interval))
        padding_after_time = pd.Series(np.arange(end_time, end_time + padding_time, interpolation_interval))
        
        padding_before = pd.DataFrame({
            'Time (s)': padding_before_time, 
            'Current (A)': [0] * len(padding_before_time)
        })
        padding_after = pd.DataFrame({
            'Time (s)': padding_after_time, 
            'Current (A)': [0] * len(padding_after_time)
        })
        
        # Combine block with padding
        block_with_padding = pd.concat([padding_before, block_df, padding_after], ignore_index=True)
        
        # Normalize time for this block
        if len(padding_before_time) > 0:
            block_with_padding['Time (s)'] = block_with_padding['Time (s)'] - padding_before_time.min()
        
        dfs_block_with_padding.append(block_with_padding)
    
    return dfs_block_with_padding


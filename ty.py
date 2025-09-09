import tkinter as tk
from tkinter import ttk, filedialog
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
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


interpolation_interval = 0.120 #s
block_interval = 7 #s
padding_time = 1.56 #s

class CSVProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Processor")
        self.root.geometry("2000x1000")
        
        # Variables
        self.input_path_var = tk.StringVar()
        self.damage_label_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        
        self.setup_ui()

    def setup_ui(self):
        # Main Frame 
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights 
        self.root.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Input Directory Selection 
        ttk.Label(main_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.input_path_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input_directory).grid(row=0, column=2, padx=5)

        # Damage Label Input
        ttk.Label(main_frame, text="Damage Label:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.damage_label_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        # Output Folder Selection 
        ttk.Label(main_frame, text="Folder name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_folder_var, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)

        # Process Button
        ttk.Button(main_frame, text="Process CSV Files", command=self.process_files).grid(row=3, column=1, pady=20)

    def browse_input_directory(self):
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_path_var.set(directory)

    def process_files(self):
        input_path = self.input_path_var.get()
        damage_label = self.damage_label_var.get()
        output_folder_name = self.output_folder_var.get() 
        base_dir = os.getcwd()
        output_folder = os.path.join(base_dir, "data", output_folder_name)
        os.makedirs(output_folder, exist_ok=True)

        process_directory(input_path, damage_label, output_folder)


def process_directory(path, damage_label, output_directory):
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.csv'):
                filepath = os.path.join(path, filename)
                try:
                    processed_df = process_csv_file(filepath)
                    write_csv(processed_df, damage_label, output_directory)
                except Exception as e:
                    print(f"An error occurred while processing {filename}: {e}")

def find_next_file_index(label, subdir):
    # Check existing files and find the next index
    existing_files = os.listdir(subdir)
    if existing_files:
        max_index = max(int(file.split('_')[-1].split('.')[0]) for file in existing_files if file.startswith(label))
        next_index = max_index + 1
    else:
        next_index = 1
    return next_index

def write_csv(processed_df, label, output_directory):
    subdir = os.path.join(output_directory, label)
    os.makedirs(subdir, exist_ok=True)

    next_index = find_next_file_index(label, subdir)

    for i, df in enumerate(processed_df):
        for j in range(5):
            filename = f"{label}_{next_index + i*5 +j}.csv"
            file_path = os.path.join(subdir, filename)

            label_df = pd.DataFrame([[label]], columns=[df.columns[0]])

            result_df = pd.concat([label_df, df], ignore_index=True)
            result_df.to_csv(file_path, index=False)
            print(f"File written to:{file_path}")

def process_csv_file(file_path):
    df = pd.read_csv(file_path, header=None, encoding='utf-8', skiprows=1, usecols=range(1,3))
    df.columns = ['DMM-1 Time (s)', 'DMM-1 Current (A)']

    df['DMM-1 Time (s)'] = df['DMM-1 Time (s)'] - df['DMM-1 Time (s)'].min()

    # interpolating the function
    interpolation_func = interp1d(df['DMM-1 Time (s)'], df['DMM-1 Current (A)'], kind='linear', fill_value='interpolate')
    max_time = df['DMM-1 Time (s)'].max()
    min_time = df['DMM-1 Time (s)'].min()
    new_time = np.arange(min_time, max_time, interpolation_interval)
    new_current = interpolation_func(new_time)

    interpolated_df = pd.DataFrame({'Time (s)': new_time, 'Current (A)': new_current})

    # Determine the number of 10 second blocks
    num_blocks = int(np.floor(max_time/block_interval))
    time_intervals = [(i*block_interval, (i+1)*block_interval) for i in range(num_blocks)]
    dfs_block_with_padding= []

    for start_time, end_time in time_intervals:
        block_df = interpolated_df[(interpolated_df['Time (s)'] >= start_time) & (interpolated_df['Time (s)'] <= end_time)]

        # Add padding
        padding_before_time = pd.Series(np.arange(start_time-padding_time, start_time, interpolation_interval))
        padding_after_time = pd.Series(np.arange(end_time, end_time+padding_time, interpolation_interval))
        padding_before = pd.DataFrame({'Time (s)': padding_before_time, 'Current (A)': [0]*len(padding_before_time)})
        padding_after = pd.DataFrame({'Time (s)': padding_after_time, 'Current (A)': [0]*len(padding_after_time)})

        block_with_padding = pd.concat([padding_before, block_df, padding_after], ignore_index=True)

        block_with_padding['Time (s)'] = block_with_padding['Time (s)'] - padding_before_time.min()

        dfs_block_with_padding.append(block_with_padding)

    return dfs_block_with_padding


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVProcessorGUI(root)
    root.mainloop()


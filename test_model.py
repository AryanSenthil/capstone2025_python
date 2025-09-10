import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import numpy as np
import tensorflow as tf
from keras import layers, models
import pandas as pd
from scipy.interpolate import interp1d
import time
from datetime import datetime
import logging
from pathlib import Path
import os

sampling_rate = 1600  #Hz
train_size = 0.8
time_period = 10
test_size = 0.1
valid_size = 0.1
random_state = 42
EPOCHS = 10


# Configuration constants
interpolation_interval = 0.120  # s
block_interval = 7  # s
padding_time = 1.56  # s


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



def write_csv_test(processed_df, test_directory):
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Combine the label with the current time
    subdir_label = f"test_{current_time_str}"

    # Create the subdirectory path with the new label
    subdir = os.path.join(test_directory, subdir_label)
    os.makedirs(subdir, exist_ok=True)
    filename = f"test_{current_time_str}"
    file_path = os.path.join(subdir, filename)
    
    for i, df in enumerate(processed_df):
        filename = f"test_{i}.csv"
        file_path = os.path.join(subdir, filename)
        df.to_csv(file_path, index=False)
    return subdir 


def process_test(testfile, test_directory):
    processed_df = process_csv_file(file_path=testfile)
    test_subdirc = write_csv_test(processed_df = processed_df, test_directory=test_directory)
    return test_subdirc


def get_test_data(filepath):
    df = pd.read_csv(filepath, header=None, encoding='utf-8')
    time_current_data = df.iloc[1:].astype(float).values
    time_current_pairs = [tuple(row) for row in time_current_data]
    return (time_current_pairs)


def interpolate_test_data(test_data, interval, period):
    interpolated_test_data = []
    
    for time_current_pairs in test_data:
        time, current = zip(*time_current_pairs)

        interpolation_func = interp1d(time, current, kind='linear', fill_value='interpolate')
        max_time = interval * (round(period/interval))
        new_time = np.arange(0, max_time, interval)
        new_time = np.arange(0, max_time, interval)
        new_current = interpolation_func(new_time)
        
        new_time_current = list(zip(new_time, new_current))
        interpolated_test_data.append(new_time_current)
    return interpolated_test_data


def read_test_csv_files(path, sampling_rate, period):
    all_test_data = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.endswith('.csv'):
                filepath = os.path.join(path, filename)
                all_test_data.append(get_test_data(filepath))
    else:
        raise ValueError(f"The path provided is neither a CSV file nor a directory: {path}")
    interval = 1/sampling_rate
    data_for_testing = interpolate_test_data(all_test_data, interval, period)
    return data_for_testing


def test_extract_data(data_list):
    new_data = []
    for time_current in data_list:
        current_data = [current for _, current in time_current]
        new_data.append(current_data)
    return new_data

def test_normalize_data(data):
    max_val = np.max((np.abs(data)))
    return data / max_val

def test_convert_to_wave(normalized_data, time_interval):
    sample_rate = tf.cast(int(1 / time_interval), tf.int32)

    audio_tensor = tf.convert_to_tensor(normalized_data, dtype=tf.float32)
    audio_tensor = tf.reshape(audio_tensor, (-1, 1))  # Ensure correct shape for audio encoding
    return tf.audio.encode_wav(audio_tensor, sample_rate)

def test_wav_generator(data, sampling_rate):
    extracted_data = test_extract_data(data)
    # Normalize the data
    normalized_data = [test_normalize_data(current_sequence) for current_sequence in extracted_data]
    # Generate WAV files
    time_interval = 1 / sampling_rate
    wav_files = [test_convert_to_wave(current_sequence, time_interval) for current_sequence in normalized_data]
    return wav_files


def save_test_wav_files(wav_file_list, base_folder_path):
    # Ensure the base folder path is unique
    version = 1
    original_path = base_folder_path
    while os.path.exists(base_folder_path):
        base_folder_path = f"{original_path}_v{version}"
        version += 1
    
    os.makedirs(base_folder_path, exist_ok=True)
    
    # Save each audio tensor to a file
    for index, audio_tensor in enumerate(wav_file_list, start=1):
        wav_filename = f"audio_{index}.wav"
        wav_filepath = os.path.join(base_folder_path, wav_filename)
        
        # Fixed: audio_tensor is already the encoded WAV data, not a tuple
        tf.io.write_file(wav_filepath, audio_tensor)
    
    return base_folder_path



def classify_audio_files(model, test_directory):
    """
    Classify all audio files in a directory using the provided model.
    
    Args:
        model: Loaded TensorFlow model
        test_directory: Path to directory containing audio files
        
    Returns:
        List of classification results
    """
    # Get all .wav files in the directory
    files_in_directory = [f for f in os.listdir(test_directory) if f.endswith('.wav')]
    
    # Sort files to ensure consistent ordering (audio_1, audio_2, etc.)
    files_in_directory.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    results = []
    
    for file_name in files_in_directory:
        # Create full file path
        file_path = os.path.join(test_directory, file_name)
        
        try:
            # Run prediction
            result = model(tf.constant(str(file_path)))
            
            # Extract classification
            class_name = result['class_names'].numpy()[0].decode('utf-8')
            class_id = result['class_ids'].numpy()[0]
            confidence_scores = result['predictions'].numpy()[0]
            
            # Store result
            classification_result = {
                'file_name': file_name,
                'file_path': file_path,
                'predicted_class': class_name,
                'class_id': class_id,
                'confidence_scores': confidence_scores.tolist()  # Convert to list for easier handling
            }
            
            results.append(classification_result)
            
            # Print result as requested
            file_number = file_name.split('_')[1].split('.')[0]  # Extract number from "audio_X.wav"
            print(f"testfile {file_number} classified as {class_name}")
            
        except Exception as e:
            print(f"Error classifying {file_name}: {e}")
            results.append({
                'file_name': file_name,
                'file_path': file_path,
                'predicted_class': 'ERROR',
                'class_id': -1,
                'confidence_scores': [],
                'error': str(e)
            })
    
    return results



def classify_test_file(test_file_path, model_path):
    """
    Complete pipeline to classify a test file using a saved model.
    
    Args:
        test_file_path: Path to the CSV test file
        model_path: Path to the saved model directory
        
    Returns:
        List of classification results
    """
    try:
        # Load the model
        model = tf.saved_model.load(model_path)
        
        # Process the test file
        test_directory = os.path.join(os.getcwd(), 'test')
        test_subdirc = process_test(test_file_path, test_directory)
        
        # Read and process CSV files
        sampling_rate = 1600  # Hz
        time_period = 10
        data_for_testing = read_test_csv_files(test_subdirc, sampling_rate, period=time_period)
        
        # Generate WAV files
        test_wave_files = test_wav_generator(data_for_testing, sampling_rate)
        
        # Save WAV files
        test_wav_files_directory = os.path.join(os.getcwd(), 'test_wav_files', 'testwave')
        test_wav_files_subdirectory = save_test_wav_files(test_wave_files, test_wav_files_directory)
        
        # Classify the audio files
        results = classify_audio_files(model, test_wav_files_subdirectory)
        
        return {
            'success': True,
            'results': results,
            'wav_directory': test_wav_files_subdirectory,
            'test_directory': test_subdirc
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'results': []
        }

class AudioClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Tester")
        self.root.geometry("2000x1500")
        
        # Variables
        self.selected_model = tk.StringVar()
        self.selected_test_file = tk.StringVar()
        self.saved_models_directory = os.path.join(os.getcwd(), 'saved_models')
        
        self.setup_ui()
        self.load_saved_models()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Model Tester", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))        
        # Model Selection Section
        model_frame = ttk.LabelFrame(main_frame, text="Select Model", padding="10")
        model_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        model_frame.columnconfigure(0, weight=1)
        
        # Model listbox with scrollbar
        model_list_frame = ttk.Frame(model_frame)
        model_list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        model_list_frame.columnconfigure(0, weight=1)
        
        self.model_listbox = tk.Listbox(model_list_frame, height=6, selectmode=tk.SINGLE)
        model_scrollbar = ttk.Scrollbar(model_list_frame, orient=tk.VERTICAL, 
                                       command=self.model_listbox.yview)
        self.model_listbox.configure(yscrollcommand=model_scrollbar.set)
        
        self.model_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        model_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Refresh models button
        refresh_btn = ttk.Button(model_frame, text="Refresh Models", 
                                command=self.load_saved_models)
        refresh_btn.grid(row=1, column=0, pady=(10, 0), sticky=tk.W)
        
        # Test File Selection Section
        file_frame = ttk.LabelFrame(main_frame, text="Select Test File", padding="10")
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="Test File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.selected_test_file, 
                                   state="readonly")
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_test_file)
        browse_btn.grid(row=0, column=2, sticky=tk.W)
        
        # Control Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        self.classify_btn = ttk.Button(button_frame, text="Test Model", 
                                      command=self.start_classification, 
                                      style="Accent.TButton")
        self.classify_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_btn = ttk.Button(button_frame, text="Clear Results", 
                                   command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Results Section
        results_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="10")
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80, 
                                                     wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def load_saved_models(self):
        """Load available models from the saved_models directory"""
        self.model_listbox.delete(0, tk.END)
        
        if not os.path.exists(self.saved_models_directory):
            self.update_status("Saved models directory not found")
            return
        
        try:
            models = [d for d in os.listdir(self.saved_models_directory) 
                     if os.path.isdir(os.path.join(self.saved_models_directory, d))]
            
            if not models:
                self.update_status("No models found in saved_models directory")
                return
            
            for model in sorted(models):
                self.model_listbox.insert(tk.END, model)
            
            self.update_status(f"Found {len(models)} model(s)")
            
        except Exception as e:
            self.update_status(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def browse_test_file(self):
        """Open file dialog to select test CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Test CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.selected_test_file.set(file_path)
            self.update_status(f"Selected test file: {os.path.basename(file_path)}")
    
    def get_selected_model(self):
        """Get the currently selected model"""
        selection = self.model_listbox.curselection()
        if not selection:
            return None
        return self.model_listbox.get(selection[0])
    
    def start_classification(self):
        """Start the classification process in a separate thread"""
        # Validate inputs
        selected_model = self.get_selected_model()
        test_file = self.selected_test_file.get()
        
        if not selected_model:
            messagebox.showwarning("Warning", "Please select a model")
            return
        
        if not test_file:
            messagebox.showwarning("Warning", "Please select a test file")
            return
        
        if not os.path.exists(test_file):
            messagebox.showerror("Error", "Test file does not exist")
            return
        
        # Disable the classify button and start progress
        self.classify_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.update_status("Classification in progress...")
        
        # Start classification in a separate thread
        model_path = os.path.join(self.saved_models_directory, selected_model)
        thread = threading.Thread(target=self.run_classification, 
                                 args=(test_file, model_path))
        thread.daemon = True
        thread.start()
    
    def run_classification(self, test_file, model_path):
        """Run the classification process"""
        try:
            # Run the classification
            result = classify_test_file(test_file, model_path)
            
            # Update UI in main thread
            self.root.after(0, self.classification_complete, result)
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'results': []
            }
            self.root.after(0, self.classification_complete, error_result)
    
    def classification_complete(self, result):
        """Handle classification completion"""
        # Stop progress and re-enable button
        self.progress.stop()
        self.classify_btn.config(state=tk.NORMAL)
        
        if result['success']:
            self.display_results(result)
            self.update_status(f"Classification complete. {len(result['results'])} files processed.")
        else:
            self.update_status(f"Classification failed: {result['error']}")
            messagebox.showerror("Classification Error", f"Failed to classify: {result['error']}")
    
    def display_results(self, result):
        """Display classification results in the text widget"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Add header
        header = f"Classification Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Test File: {os.path.basename(self.selected_test_file.get())}\n"
        header += f"Model: {self.get_selected_model()}\n"
        header += f"Total Files Processed: {len(result['results'])}\n"
        header += "-" * 80 + "\n\n"
        
        self.results_text.insert(tk.END, header)
        
        # Add individual results
        for i, res in enumerate(result['results'], 1):
            if res['predicted_class'] != 'ERROR':
                result_text = f"File {i}: {res['predicted_class']}\n"
            else:
                result_text = f"File {i}: ERROR - {res.get('error', 'Unknown error')}\n"
            
            self.results_text.insert(tk.END, result_text)
        
        # Add summary
        successful_classifications = [r for r in result['results'] if r['predicted_class'] != 'ERROR']
        if successful_classifications:
            class_counts = {}
            for res in successful_classifications:
                class_name = res['predicted_class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            summary = "\nSummary:\n"
            for class_name, count in class_counts.items():
                summary += f"  {class_name}: {count} file(s)\n"
            
            self.results_text.insert(tk.END, summary)
        
        
        self.results_text.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear the results text widget"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)
        self.update_status("Results cleared")
    
    def update_status(self, message):
        """Update the status bar"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = AudioClassifierGUI(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()


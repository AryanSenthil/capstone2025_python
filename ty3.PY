import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import numpy as np
import tensorflow as tf
from keras import layers, models
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import json
import time
from pathlib import Path

# Global parameters
sampling_rate = 1600  # Hz
train_size = 0.8
time_period = 10
test_size = 0.1
valid_size = 0.1
random_state = 42
EPOCHS = 10

class ProgressCallback:
    """Callback class for progress tracking"""
    def __init__(self, gui_callback=None):
        self.gui_callback = gui_callback
        
    def update_progress(self, stage, progress, message=""):
        if self.gui_callback:
            self.gui_callback(stage, progress, message)

def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio,label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)

def get_audio_length(directory):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(subdir_path, filename)
                    audio_binary = tf.io.read_file(file_path)
                    waveform, sample_rate = tf.audio.decode_wav(audio_binary)
                    return waveform.shape[0], sample_rate
    return None # case of no audio files 

def num_files(directory):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        audio_files = [f for f in os.listdir(subdir_path) if f.endswith('.wav')]
        num_files = len(audio_files)
        if num_files > 0:
            return num_files
        else:
            return None

def save_wav_files(wav_files, base_folder_path, progress_callback=None):
    # create a new version of the directory if it already exists
    version = 1
    while os.path.exists(base_folder_path):
        base_folder_path = base_folder_path.rstrip('/')+f"_v{version}"
        version += 1
    
    total_files = len(wav_files)
    
    # Create a subdirectory for each unique class and save WAV files 
    for i, (audio_binary, damage_type) in enumerate(wav_files):
        class_directory = os.path.join(base_folder_path, damage_type)

        # create a directory if it doesn't exist
        os.makedirs(class_directory, exist_ok=True)

        # Find the next available file index
        existing_files = os.listdir(class_directory)
        existing_indices = [int(f.split('_')[-1].split('.')[0]) for f in existing_files if f.endswith('.wav')]
        file_index = max(existing_indices, default=0) + 1

        # Define the full path for the new WAV file
        wav_filename = f"{damage_type}_audio_{file_index}.wav"
        wav_filepath = os.path.join(class_directory, wav_filename)

        # write and save the WAV file 
        tf.io.write_file(wav_filepath, audio_binary)
        
        # Update progress
        if progress_callback:
            progress = (i + 1) / total_files * 100
            progress_callback.update_progress("saving_wav", progress, f"Saving audio file {i+1}/{total_files}")
    
    return base_folder_path

def extract_data_and_type(data_list):
    new_data = []
    for time_current, damage_type in data_list:
        current_data = [a for _, a in time_current]
        new_data.append((current_data, damage_type))
    return new_data

def normalize_data(data):
    max_val = np.max((np.abs(data)))
    return data / max_val

def convert_to_wave(normalized_data, time_interval):
    sample_rate = tf.cast(int(1/time_interval), tf.int32)

    audio_tensor = tf.convert_to_tensor(normalized_data, dtype=tf.float32)
    audio_tensor = tf.reshape(audio_tensor, (-1,1)) # Ensure correct shape for audio encoding
    return tf.audio.encode_wav(audio_tensor, sample_rate)

def wav_generator(data, sampling_rate, progress_callback=None):
    classification_data = extract_data_and_type(data)
    
    if progress_callback:
        progress_callback.update_progress("processing", 25, "Normalizing data...")
    
    # Normalize the data
    normalized_data = [(normalize_data(current_sequence), damage_type) for current_sequence, damage_type in classification_data]
    
    if progress_callback:
        progress_callback.update_progress("processing", 50, "Generating WAV files...")
    
    # Generate WAV files
    time_interval = 1 / sampling_rate
    wav_files = [(convert_to_wave(current_sequence,
                                  time_interval), damage_type)
                 for current_sequence, damage_type in normalized_data]
    return wav_files

def process_csv_file(file_path):
    # Function to process a single csv file
    df = pd.read_csv(file_path, header=None, encoding='utf-8')
    
    # Check if the dataframe has at least 2 rows (index 0 and 1)
    if len(df) < 2:
        raise ValueError(f"CSV file {file_path} must have at least 2 rows to contain type_of_damage at row 1")
    
    # Check if row 1 has at least 1 column (index 0)
    if len(df.columns) < 1 or pd.isna(df.iloc[1, 0]):
        raise ValueError(f"CSV file {file_path} is missing type_of_damage value at position [1,0]")
    
    type_of_damage = df.iloc[1, 0]
    time_current_data = df.iloc[2:].astype(float).values
    time_current_pairs = [tuple(row) for row in time_current_data]
    return (time_current_pairs, type_of_damage)

def read_csv_files(path, sampling_rate, period, progress_callback=None):
    data_for_model = []
    all_data = [] # ((time,current), type, deformation) for each file

    if progress_callback:
        progress_callback.update_progress("loading", 10, f"Reading CSV files from {path}...")

    if os.path.isdir(path):
        csv_files = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('.csv'):
                    csv_files.append(os.path.join(root, filename))
        
        for i, file_path in enumerate(csv_files):
            all_data.append(process_csv_file(file_path))
            if progress_callback:
                progress = 10 + (i + 1) / len(csv_files) * 30
                progress_callback.update_progress("loading", progress, f"Processing file {i+1}/{len(csv_files)}")
                
    elif os.path.isfile(path) and path.endswith('.csv'):
        all_data.append(process_csv_file(path))
    else:
        raise ValueError(f"The path provided is neither a CSV file nor a directory: {path}")

    if progress_callback:
        progress_callback.update_progress("loading", 50, "Interpolating data...")

    interval = 1/sampling_rate
    data_for_model = interpolate_data(all_data, interval, period)
    return data_for_model

def interpolate_data(data, interval, period):
    interpolated_data = []

    for time_current_pairs, type_of_damage in data:
        # Extract the time and voltage values
        time, current = zip(*time_current_pairs)

        interpolation_func = interp1d(time, current, kind='linear', fill_value='interpolate')
        max_time = interval * (round(period/interval))
        new_time = np.arange(0, max_time, interval)
        new_current = interpolation_func(new_time)

        new_time_current = list(zip(new_time, new_current))
        interpolated_data.append((new_time_current, type_of_damage))
    return interpolated_data

def generate_model_evaluation_pdf(model, history, test_spectrogram_ds, label_names, 
                                model_path=None, output_path=None, figsize=(12, 8)):
    """
    Generate a comprehensive PDF report with model evaluation metrics and plots.
    """
    
    # Generate output filename if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"model_evaluation_report_{timestamp}.pdf"
    
    # Evaluate model and get predictions
    print("Evaluating model...")
    eval_results = model.evaluate(test_spectrogram_ds, return_dict=True, verbose=0)
    
    print("Generating predictions...")
    y_pred = model.predict(test_spectrogram_ds, verbose=0)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)
    
    # Calculate confusion matrix
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    
    # Create PDF with multiple pages
    with PdfPages(output_path) as pdf:
        
        # Page 1: Model Summary and Evaluation Metrics
        fig = plt.figure(figsize=figsize)
        fig.suptitle('Model Evaluation Report', fontsize=16, fontweight='bold')
        
        # Remove axes and add text summary
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        # Prepare summary text
        summary_text = f"""
Model Evaluation Summary
{'=' * 50}

Model Path: {model_path if model_path else 'Not specified'}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Test Set Performance:
"""
        
        for metric, value in eval_results.items():
            if isinstance(value, float):
                summary_text += f"  {metric.capitalize()}: {value:.4f}\n"
            else:
                summary_text += f"  {metric.capitalize()}: {value}\n"
        
        summary_text += f"""
Dataset Information:
  Number of Classes: {len(label_names)}
  Class Names: {', '.join(label_names)}

Training Information:
  Total Epochs: {len(history.history['loss'])}
  Final Training Loss: {history.history['loss'][-1]:.4f}
  Final Validation Loss: {history.history['val_loss'][-1]:.4f}
  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}
  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Confusion Matrix
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle('Confusion Matrix', fontsize=16, fontweight='bold')
        
        sns.heatmap(confusion_mtx, 
                   xticklabels=label_names, 
                   yticklabels=label_names,
                   annot=True, 
                   fmt='g', 
                   cmap='Blues',
                   ax=ax)
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Add accuracy per class
        confusion_mtx_np = confusion_mtx.numpy()
        class_accuracies = np.diag(confusion_mtx_np) / np.sum(confusion_mtx_np, axis=1)
        
        # Add text box with per-class accuracies
        acc_text = "Per-Class Accuracy:\n" + "\n".join([
            f"{label}: {acc:.3f}" for label, acc in zip(label_names, class_accuracies)
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(1.02, 1, acc_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Training History - Loss and Accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss plot
        metrics = history.history
        epochs = range(1, len(metrics['loss']) + 1)
        
        ax1.plot(epochs, metrics['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, metrics['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss [CrossEntropy]', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, max(max(metrics['loss']), max(metrics['val_loss'])) * 1.1])
        
        # Accuracy plot
        ax2.plot(epochs, [x*100 for x in metrics['accuracy']], 'b-', 
                label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, [x*100 for x in metrics['val_accuracy']], 'r-', 
                label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy [%]', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 4: Class Distribution Analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Class distribution in test set
        unique, counts = np.unique(y_true.numpy(), return_counts=True)
        ax1.bar([label_names[i] for i in unique], counts, alpha=0.7, color='skyblue')
        ax1.set_title('Test Set Class Distribution')
        ax1.set_ylabel('Number of Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Prediction distribution
        unique_pred, counts_pred = np.unique(y_pred.numpy(), return_counts=True)
        ax2.bar([label_names[i] for i in unique_pred], counts_pred, alpha=0.7, color='lightcoral')
        ax2.set_title('Prediction Distribution')
        ax2.set_ylabel('Number of Predictions')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"PDF report generated successfully: {output_path}")
    return output_path

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback for training progress updates"""
    
    def __init__(self, gui_callback=None):
        self.gui_callback = gui_callback
        self.total_epochs = 0
        
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        
    def on_epoch_end(self, epoch, logs=None):
        if self.gui_callback:
            progress = (epoch + 1) / self.total_epochs * 100
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            accuracy = logs.get('accuracy', 0)
            val_accuracy = logs.get('val_accuracy', 0)
            
            message = f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, Val Acc: {val_accuracy:.4f}"
            self.gui_callback.update_progress("training", progress, message)

def train_classification_model(data_paths, model_save_name, sampling_rate=16000, time_period=10, epochs=10, progress_callback=None):
    """
    Train a classification model from CSV data and save it.
    """
    
    # Flatten data from multiple paths
    if progress_callback:
        progress_callback.update_progress("loading", 0, "Starting data loading...")
        
    data = []
    for i, path in enumerate(data_paths):
        if progress_callback:
            progress_callback.update_progress("loading", i / len(data_paths) * 40, f"Loading data from {path}...")
        data.extend(read_csv_files(path, sampling_rate, period=time_period, progress_callback=progress_callback))
    
    # Generate wav files
    if progress_callback:
        progress_callback.update_progress("processing", 0, "Generating wav files...")
        
    wav_files = wav_generator(data, sampling_rate, progress_callback)
    
    # Save wav files
    wave_folder = os.path.join(os.getcwd(), "wave_files/wave_files")
    directory = save_wav_files(wav_files, wave_folder, progress_callback)
    
    # Get audio properties
    if progress_callback:
        progress_callback.update_progress("processing", 90, "Analyzing audio files...")
        
    audio_length, sample_rate = get_audio_length(directory)
    num_files_count = num_files(directory)
    
    if audio_length is None:
        raise FileNotFoundError(f"There are no audio files in {directory}")
    else:
        print(f"The audio length in {directory} is {audio_length} and the sample rate is {sample_rate}")
    
    if num_files_count is None:
        raise FileNotFoundError(f"There is an error with the no of files in {directory}")
    else:
        print(f"The no of files is {num_files_count}")
    
    # Create datasets
    if progress_callback:
        progress_callback.update_progress("dataset", 0, "Creating TensorFlow datasets...")
        
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=directory,
        batch_size=1,
        validation_split=0.2,
        seed=0,
        output_sequence_length=audio_length,
        subset='both'
    )
    
    label_names = np.array(train_ds.class_names)
    print("Label names: ", label_names)
    
    # Squeeze extra dimensions
    def squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels
    
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
    
    # Split validation set
    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)
    
    # Create spectrogram datasets
    if progress_callback:
        progress_callback.update_progress("dataset", 50, "Creating spectrogram datasets...")
        
    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)
    
    # Get input shape
    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break
    input_shape = example_spectrograms.shape[1:]
    num_labels = len(label_names)
    
    # Optimize datasets
    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    # Create normalization layer
    norm_layer = layers.Normalization()
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))
    
    # Build model
    if progress_callback:
        progress_callback.update_progress("model", 0, "Building model architecture...")
        
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
    
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    
    # Train model
    if progress_callback:
        progress_callback.update_progress("training", 0, "Starting model training...")
        
    training_callback = TrainingProgressCallback(progress_callback)
    
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
            training_callback
        ],
    )
    
    # Test model
    if progress_callback:
        progress_callback.update_progress("evaluation", 0, "Evaluating model...")
        
    y_pred = model.predict(test_spectrogram_ds)
    
    # Create export model class
    class ExportModel(tf.Module):
        def __init__(self, model):
            self.model = model
            # Accept either a string-filename or a batch of waveforms.
            self.__call__.get_concrete_function(
                x=tf.TensorSpec(shape=(), dtype=tf.string))
            self.__call__.get_concrete_function(
                x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))
        
        @tf.function
        def __call__(self, x):
            # If they pass a string, load the file and decode it.
            if x.dtype == tf.string:
                x = tf.io.read_file(x)
                x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000)
                x = tf.squeeze(x, axis=-1)
                x = x[tf.newaxis, :]
            
            x = get_spectrogram(x)
            results = self.model(x, training=False)
            class_ids = tf.argmax(results, axis=-1)
            class_names = tf.gather(label_names, class_ids)
            return {
                'predictions': results,
                'class_ids': class_ids,
                'class_names': class_names
            }
    
    # Save model
    if progress_callback:
        progress_callback.update_progress("saving", 50, f"Saving model as '{model_save_name}'...")
        
    export_model = ExportModel(model)
    save_model_directory = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_model_directory, exist_ok=True)
    model_path = os.path.join(save_model_directory, model_save_name)
    tf.saved_model.save(export_model, model_path)
    
    # Save training configuration
    config = {
        'model_name': model_save_name,
        'data_paths': data_paths,
        'sampling_rate': sampling_rate,
        'time_period': time_period,
        'epochs': epochs,
        'label_names': label_names.tolist(),
        'training_history': {k: [float(x) for x in v] for k, v in history.history.items()},
        'created_at': datetime.now().isoformat()
    }
    
    config_path = os.path.join(save_model_directory, f"{model_save_name}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if progress_callback:
        progress_callback.update_progress("saving", 100, "Model saved successfully!")
    
    print(f"Model saved successfully to: {model_path}")
    print(f"Configuration saved to: {config_path}")
    
    return {
        'model': model,
        'history': history,
        'model_path': model_path,
        'config_path': config_path,
        'test_spectrogram_ds': test_spectrogram_ds,
        'label_names': label_names,
        'train_spectrogram_ds': train_spectrogram_ds,
        'val_spectrogram_ds': val_spectrogram_ds
    }

class EnhancedModelTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Classification Model Trainer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#50C878',
            'warning': '#FFB347',
            'error': '#FF6B6B',
            'info': '#87CEEB',
            'light': '#F8F9FA',
            'dark': '#343A40'
        }
        
        # Configure ttk styles
        self.setup_styles()
        
        # Get data path and folders
        self.current_path = os.getcwd()
        self.data_path = os.path.join(self.current_path, "data")
        
        # Check if data directory exists
        if not os.path.exists(self.data_path):
            messagebox.showerror("Error", f"Data directory not found: {self.data_path}")
            return
            
        self.folders = [folder for folder in os.listdir(self.data_path) 
                       if os.path.isdir(os.path.join(self.data_path, folder))]
        
        self.selected_folders = []
        self.training_thread = None
        self.is_training = False
        
        self.setup_ui()
        
    def setup_styles(self):
        """Setup custom ttk styles"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure('Primary.TButton', 
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        
        style.configure('Warning.TButton',
                       background=self.colors['warning'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        
        style.configure('Error.TButton',
                       background=self.colors['error'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none')
        
        # Configure progressbar styles
        style.configure('Primary.Horizontal.TProgressbar',
                       background=self.colors['primary'],
                       troughcolor='#E0E0E0',
                       borderwidth=0,
                       lightcolor=self.colors['primary'],
                       darkcolor=self.colors['primary'])
        
        style.configure('Success.Horizontal.TProgressbar',
                       background=self.colors['success'],
                       troughcolor='#E0E0E0',
                       borderwidth=0,
                       lightcolor=self.colors['success'],
                       darkcolor=self.colors['success'])
        
    def setup_ui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Training tab
        self.training_frame = ttk.Frame(notebook, padding="20")
        notebook.add(self.training_frame, text="Model Training")
        
        # Results tab
        self.results_frame = ttk.Frame(notebook, padding="20")
        notebook.add(self.results_frame, text="Training Results")
        
        # Settings tab
        self.settings_frame = ttk.Frame(notebook, padding="20")
        notebook.add(self.settings_frame, text="Settings")
        
        self.setup_training_tab()
        self.setup_results_tab()
        self.setup_settings_tab()
        
    def setup_training_tab(self):
        """Setup the main training interface"""
        # Title with icon
        title_frame = ttk.Frame(self.training_frame)
        title_frame.pack(fill='x', pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="🤖 Classification Model Trainer", 
                               font=("Arial", 18, "bold"))
        title_label.pack(side='left')
        
        # Data path selection
        path_frame = ttk.LabelFrame(self.training_frame, text="Data Directory", padding="10")
        path_frame.pack(fill='x', pady=(0, 15))
        
        self.data_path_var = tk.StringVar(value=self.data_path)
        path_entry = ttk.Entry(path_frame, textvariable=self.data_path_var, width=60)
        path_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_button = ttk.Button(path_frame, text="Browse", command=self.browse_data_directory)
        browse_button.pack(side='right')
        
        # Folder selection section
        folder_frame = ttk.LabelFrame(self.training_frame, text="Select Data Folders", padding="10")
        folder_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        # Folder selection controls
        controls_frame = ttk.Frame(folder_frame)
        controls_frame.pack(fill='x', pady=(0, 10))
        
        select_all_button = ttk.Button(controls_frame, text="Select All", 
                                     command=self.select_all_folders)
        select_all_button.pack(side='left', padx=(0, 5))
        
        clear_all_button = ttk.Button(controls_frame, text="Clear All", 
                                    command=self.clear_selection)
        clear_all_button.pack(side='left', padx=(0, 5))
        
        refresh_button = ttk.Button(controls_frame, text="Refresh", 
                                  command=self.refresh_folders)
        refresh_button.pack(side='left')
        
        # Folder info label
        self.folder_info_label = ttk.Label(controls_frame, text="", foreground=self.colors['info'])
        self.folder_info_label.pack(side='right')
        
        # Scrollable frame for checkboxes
        checkbox_canvas = tk.Canvas(folder_frame, height=200, bg='white')
        scrollbar = ttk.Scrollbar(folder_frame, orient="vertical", command=checkbox_canvas.yview)
        self.scrollable_frame = ttk.Frame(checkbox_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: checkbox_canvas.configure(scrollregion=checkbox_canvas.bbox("all"))
        )
        
        checkbox_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        checkbox_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.create_folder_checkboxes()
        
        checkbox_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configuration section
        config_frame = ttk.LabelFrame(self.training_frame, text="Training Configuration", padding="10")
        config_frame.pack(fill='x', pady=(0, 15))
        
        # Left side - Basic parameters
        left_config = ttk.Frame(config_frame)
        left_config.pack(side='left', fill='both', expand=True)
        
        # Model name
        ttk.Label(left_config, text="Model Name:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky='w', pady=2)
        self.model_name_var = tk.StringVar(value="classifier_model")
        model_name_entry = ttk.Entry(left_config, textvariable=self.model_name_var, width=25)
        model_name_entry.grid(row=0, column=1, sticky='ew', padx=(10, 0), pady=2)
        
        # Epochs
        ttk.Label(left_config, text="Epochs:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky='w', pady=2)
        self.epochs_var = tk.StringVar(value="10")
        epochs_entry = ttk.Entry(left_config, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=1, column=1, sticky='w', padx=(10, 0), pady=2)
        
        # Sampling rate
        ttk.Label(left_config, text="Sampling Rate:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky='w', pady=2)
        self.sampling_rate_var = tk.StringVar(value="1600")
        sampling_rate_entry = ttk.Entry(left_config, textvariable=self.sampling_rate_var, width=10)
        sampling_rate_entry.grid(row=2, column=1, sticky='w', padx=(10, 0), pady=2)
        
        # Time period
        ttk.Label(left_config, text="Time Period:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky='w', pady=2)
        self.time_period_var = tk.StringVar(value="10")
        time_period_entry = ttk.Entry(left_config, textvariable=self.time_period_var, width=10)
        time_period_entry.grid(row=3, column=1, sticky='w', padx=(10, 0), pady=2)
        
        left_config.columnconfigure(1, weight=1)
        
        # Right side - Options
        right_config = ttk.Frame(config_frame)
        right_config.pack(side='right', fill='both', padx=(20, 0))
        
        # Auto-generate report checkbox
        self.auto_report_var = tk.BooleanVar(value=True)
        auto_report_check = ttk.Checkbutton(right_config, text="📊 Auto-generate evaluation report", 
                                           variable=self.auto_report_var)
        auto_report_check.pack(anchor='w', pady=2)
        
        # Save training data checkbox
        self.save_data_var = tk.BooleanVar(value=True)
        save_data_check = ttk.Checkbutton(right_config, text="💾 Save training data", 
                                        variable=self.save_data_var)
        save_data_check.pack(anchor='w', pady=2)
        
        # Open results automatically
        self.auto_open_var = tk.BooleanVar(value=True)
        auto_open_check = ttk.Checkbutton(right_config, text="🔗 Open results automatically", 
                                        variable=self.auto_open_var)
        auto_open_check.pack(anchor='w', pady=2)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.training_frame, text="Training Progress", padding="10")
        progress_frame.pack(fill='x', pady=(0, 15))
        
        # Main progress bar
        self.main_progress = ttk.Progressbar(progress_frame, style='Primary.Horizontal.TProgressbar', 
                                           mode='determinate', length=400)
        self.main_progress.pack(fill='x', pady=(0, 5))
        
        # Stage-specific progress bars
        stages_frame = ttk.Frame(progress_frame)
        stages_frame.pack(fill='x')
        
        # Create progress bars for different stages
        self.stage_progress_bars = {}
        self.stage_labels = {}
        
        stages = ['loading', 'processing', 'dataset', 'model', 'training', 'evaluation', 'saving']
        stage_names = ['Loading Data', 'Processing', 'Dataset Creation', 'Model Building', 
                      'Training', 'Evaluation', 'Saving']
        
        for i, (stage, name) in enumerate(zip(stages, stage_names)):
            frame = ttk.Frame(stages_frame)
            frame.pack(fill='x', pady=1)
            
            label = ttk.Label(frame, text=f"{name}:", width=15, anchor='w')
            label.pack(side='left')
            
            progress_bar = ttk.Progressbar(frame, mode='determinate', length=200)
            progress_bar.pack(side='left', fill='x', expand=True, padx=(5, 0))
            
            self.stage_progress_bars[stage] = progress_bar
            self.stage_labels[stage] = label
        
        # Status and message
        self.status_label = ttk.Label(progress_frame, text="Ready to train", 
                                    foreground=self.colors['success'], font=("Arial", 10, "bold"))
        self.status_label.pack(pady=(10, 0))
        
        self.message_label = ttk.Label(progress_frame, text="", 
                                     foreground=self.colors['info'], font=("Arial", 9))
        self.message_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(self.training_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        self.train_button = ttk.Button(button_frame, text="🚀 Start Training", 
                                     style='Primary.TButton', command=self.start_training)
        self.train_button.pack(side='left', padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹️ Stop Training", 
                                    style='Error.TButton', command=self.stop_training, 
                                    state='disabled')
        self.stop_button.pack(side='left', padx=(0, 10))
        
        # Estimated time label
        self.time_estimate_label = ttk.Label(button_frame, text="", 
                                           foreground=self.colors['info'])
        self.time_estimate_label.pack(side='right')
        
    def setup_results_tab(self):
        """Setup the results viewing interface"""
        # Title
        title_label = ttk.Label(self.results_frame, text="📈 Training Results", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Results display area
        results_notebook = ttk.Notebook(self.results_frame)
        results_notebook.pack(fill='both', expand=True)
        
        # Training history tab
        history_frame = ttk.Frame(results_notebook, padding="10")
        results_notebook.add(history_frame, text="Training History")
        
        # Model info tab
        model_frame = ttk.Frame(results_notebook, padding="10")
        results_notebook.add(model_frame, text="Model Information")
        
        # Files tab
        files_frame = ttk.Frame(results_notebook, padding="10")
        results_notebook.add(files_frame, text="Generated Files")
        
        # Training history text area
        self.history_text = scrolledtext.ScrolledText(history_frame, height=20, width=80)
        self.history_text.pack(fill='both', expand=True)
        
        # Model info text area
        self.model_info_text = scrolledtext.ScrolledText(model_frame, height=20, width=80)
        self.model_info_text.pack(fill='both', expand=True)
        
        # Files list
        files_list_frame = ttk.Frame(files_frame)
        files_list_frame.pack(fill='both', expand=True)
        
        self.files_listbox = tk.Listbox(files_list_frame, height=15)
        files_scrollbar = ttk.Scrollbar(files_list_frame, orient="vertical", 
                                      command=self.files_listbox.yview)
        self.files_listbox.configure(yscrollcommand=files_scrollbar.set)
        
        self.files_listbox.pack(side="left", fill="both", expand=True)
        files_scrollbar.pack(side="right", fill="y")
        
        # File operations buttons
        file_buttons_frame = ttk.Frame(files_frame)
        file_buttons_frame.pack(fill='x', pady=(10, 0))
        
        open_file_button = ttk.Button(file_buttons_frame, text="Open Selected", 
                                    command=self.open_selected_file)
        open_file_button.pack(side='left', padx=(0, 5))
        
        open_folder_button = ttk.Button(file_buttons_frame, text="Open Folder", 
                                      command=self.open_results_folder)
        open_folder_button.pack(side='left')
        
    def setup_settings_tab(self):
        """Setup the settings interface"""
        # Title
        title_label = ttk.Label(self.settings_frame, text="⚙️ Settings", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Model architecture settings
        arch_frame = ttk.LabelFrame(self.settings_frame, text="Model Architecture", padding="10")
        arch_frame.pack(fill='x', pady=(0, 15))
        
        # Advanced training settings
        advanced_frame = ttk.LabelFrame(self.settings_frame, text="Advanced Settings", padding="10")
        advanced_frame.pack(fill='x', pady=(0, 15))
        
        # Export/Import settings
        export_frame = ttk.LabelFrame(self.settings_frame, text="Configuration", padding="10")
        export_frame.pack(fill='x')
        
        # Save/Load configuration buttons
        config_buttons_frame = ttk.Frame(export_frame)
        config_buttons_frame.pack(fill='x')
        
        save_config_button = ttk.Button(config_buttons_frame, text="💾 Save Configuration", 
                                      command=self.save_configuration)
        save_config_button.pack(side='left', padx=(0, 5))
        
        load_config_button = ttk.Button(config_buttons_frame, text="📁 Load Configuration", 
                                      command=self.load_configuration)
        load_config_button.pack(side='left')
        
    def create_folder_checkboxes(self):
        """Create checkboxes for each folder"""
        # Clear existing checkboxes
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.folder_vars = {}
        for i, folder in enumerate(self.folders):
            var = tk.BooleanVar()
            self.folder_vars[folder] = var
            
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(fill='x', padx=5, pady=2)
            
            checkbox = ttk.Checkbutton(frame, text=folder, variable=var, 
                                     command=self.update_folder_info)
            checkbox.pack(side='left')
            
            # Add folder info (number of CSV files)
            folder_path = os.path.join(self.data_path, folder)
            if os.path.exists(folder_path):
                csv_count = sum(1 for f in os.listdir(folder_path) if f.endswith('.csv'))
                info_label = ttk.Label(frame, text=f"({csv_count} CSV files)", 
                                     foreground=self.colors['info'])
                info_label.pack(side='right')
        
        self.update_folder_info()
        
    def update_folder_info(self):
        """Update folder selection info"""
        selected_count = sum(1 for var in self.folder_vars.values() if var.get())
        total_count = len(self.folder_vars)
        self.folder_info_label.config(text=f"Selected: {selected_count}/{total_count}")
        
    def browse_data_directory(self):
        """Browse for data directory"""
        directory = filedialog.askdirectory(initialdir=self.data_path)
        if directory:
            self.data_path = directory
            self.data_path_var.set(directory)
            self.refresh_folders()
            
    def refresh_folders(self):
        """Refresh the folder list"""
        if os.path.exists(self.data_path):
            self.folders = [folder for folder in os.listdir(self.data_path) 
                           if os.path.isdir(os.path.join(self.data_path, folder))]
            self.create_folder_checkboxes()
        else:
            messagebox.showerror("Error", f"Directory does not exist: {self.data_path}")
            
    def select_all_folders(self):
        """Select all folders"""
        for var in self.folder_vars.values():
            var.set(True)
        self.update_folder_info()
        
    def clear_selection(self):
        """Clear all folder selections"""
        for var in self.folder_vars.values():
            var.set(False)
        self.update_folder_info()
        
    def estimate_training_time(self):
        """Estimate training time based on selected data and epochs"""
        selected_folders = [folder for folder, var in self.folder_vars.items() if var.get()]
        if not selected_folders:
            return ""
        
        # Simple estimation based on number of files and epochs
        total_files = 0
        for folder in selected_folders:
            folder_path = os.path.join(self.data_path, folder)
            if os.path.exists(folder_path):
                total_files += sum(1 for f in os.listdir(folder_path) if f.endswith('.csv'))
        
        try:
            epochs = int(self.epochs_var.get())
            estimated_minutes = (total_files * epochs * 0.1)  # Rough estimate
            if estimated_minutes < 1:
                return "Est. time: < 1 minute"
            elif estimated_minutes < 60:
                return f"Est. time: ~{int(estimated_minutes)} minutes"
            else:
                hours = int(estimated_minutes // 60)
                minutes = int(estimated_minutes % 60)
                return f"Est. time: ~{hours}h {minutes}m"
        except:
            return ""
    
    def update_progress(self, stage, progress, message=""):
        """Update progress bars and status"""
        def _update():
            # Update main progress (average of all stages)
            total_progress = sum(bar['value'] for bar in self.stage_progress_bars.values())
            avg_progress = total_progress / len(self.stage_progress_bars)
            self.main_progress['value'] = avg_progress
            
            # Update specific stage
            if stage in self.stage_progress_bars:
                self.stage_progress_bars[stage]['value'] = progress
                
                # Color code the stage label based on progress
                label = self.stage_labels[stage]
                if progress >= 100:
                    label.config(foreground=self.colors['success'])
                elif progress > 0:
                    label.config(foreground=self.colors['primary'])
                else:
                    label.config(foreground=self.colors['dark'])
            
            # Update status and message
            if message:
                self.message_label.config(text=message)
            
            # Update time estimate
            time_est = self.estimate_training_time()
            self.time_estimate_label.config(text=time_est)
            
        self.root.after(0, _update)
    
    def validate_inputs(self):
        """Validate all input parameters"""
        # Check if any folders are selected
        selected_folders = [folder for folder, var in self.folder_vars.items() if var.get()]
        if not selected_folders:
            messagebox.showerror("Error", "Please select at least one data folder")
            return False
        
        # Check model name
        model_name = self.model_name_var.get().strip()
        if not model_name:
            messagebox.showerror("Error", "Please enter a model name")
            return False
        
        # Validate epochs
        try:
            epochs = int(self.epochs_var.get())
            if epochs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Epochs must be a positive integer")
            return False
        
        # Validate sampling rate
        try:
            sampling_rate = int(self.sampling_rate_var.get())
            if sampling_rate <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Sampling rate must be a positive integer")
            return False
        
        # Validate time period
        try:
            time_period = float(self.time_period_var.get())
            if time_period <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Time period must be a positive number")
            return False
        
        return True
    
    def start_training(self):
        """Start the training process"""
        if not self.validate_inputs():
            return
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress")
            return
        
        # Reset progress bars
        for bar in self.stage_progress_bars.values():
            bar['value'] = 0
        self.main_progress['value'] = 0
        
        # Reset stage label colors
        for label in self.stage_labels.values():
            label.config(foreground=self.colors['dark'])
        
        # Update UI state
        self.train_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.is_training = True
        
        self.status_label.config(text="🚀 Training started...", foreground=self.colors['primary'])
        
        # Create progress callback
        progress_callback = ProgressCallback(self.update_progress)
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.train_model, args=(progress_callback,))
        self.training_thread.daemon = True
        self.training_thread.start()
    
    def stop_training(self):
        """Stop the training process"""
        if self.is_training and self.training_thread:
            self.is_training = False
            self.status_label.config(text="⏹️ Training stopped by user", foreground=self.colors['warning'])
            self.train_button.config(state='normal')
            self.stop_button.config(state='disabled')
            messagebox.showinfo("Info", "Training has been stopped")
    
    def train_model(self, progress_callback):
        """Main training function"""
        try:
            # Get parameters
            selected_folders = [folder for folder, var in self.folder_vars.items() if var.get()]
            data_paths = [os.path.join(self.data_path, folder) for folder in selected_folders]
            
            model_name = self.model_name_var.get().strip()
            epochs = int(self.epochs_var.get())
            sampling_rate = int(self.sampling_rate_var.get())
            time_period = float(self.time_period_var.get())
            
            auto_report = self.auto_report_var.get()
            auto_open = self.auto_open_var.get()
            
            # Start training
            results = train_classification_model(
                data_paths=data_paths,
                model_save_name=model_name,
                sampling_rate=sampling_rate,
                time_period=time_period,
                epochs=epochs,
                progress_callback=progress_callback
            )
            
            if not self.is_training:  # Check if training was stopped
                return
            
            model = results['model']
            history = results['history']
            model_path = results['model_path']
            config_path = results['config_path']
            test_spectrogram_ds = results['test_spectrogram_ds']
            label_names = results['label_names']
            
            # Generate evaluation report if requested
            report_path = None
            if auto_report:
                progress_callback.update_progress("evaluation", 50, "Generating evaluation report...")
                
                # Create report directory if it doesn't exist
                report_dir = os.path.join(os.getcwd(), "reports")
                os.makedirs(report_dir, exist_ok=True)
                
                # Generate report path
                report_filename = f"{model_name}_evaluation_report.pdf"
                report_path = os.path.join(report_dir, report_filename)
                
                # Generate the PDF report
                report_path = generate_model_evaluation_pdf(
                    model=model,
                    history=history,
                    test_spectrogram_ds=test_spectrogram_ds,
                    label_names=label_names,
                    model_path=model_path,
                    output_path=report_path
                )
                
                progress_callback.update_progress("evaluation", 100, "Evaluation report generated!")
            
            # Training completed successfully
            self.root.after(0, lambda: self.training_completed(results, report_path, auto_open))
            
        except Exception as e:
            if self.is_training:  # Only show error if training wasn't stopped
                self.root.after(0, lambda: self.training_failed(str(e)))
    
    def training_completed(self, results, report_path=None, auto_open=False):
        """Handle successful training completion"""
        self.is_training = False
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        model_path = results['model_path']
        config_path = results['config_path']
        
        # Update status
        self.status_label.config(text="✅ Training completed successfully!", 
                               foreground=self.colors['success'])
        
        # Update results tabs
        self.update_results_display(results, report_path)
        
        # Create success message
        message_parts = [f"Model trained successfully!\n\nModel saved to: {model_path}"]
        
        if config_path:
            message_parts.append(f"Configuration saved to: {config_path}")
        
        if report_path:
            message_parts.append(f"Evaluation report saved to: {report_path}")
        
        message = "\n".join(message_parts)
        
        # Show success dialog
        messagebox.showinfo("🎉 Training Complete", message)
        
        # Auto-open results if requested
        if auto_open:
            self.auto_open_results(model_path, report_path)
    
    def training_failed(self, error_message):
        """Handle training failure"""
        self.is_training = False
        self.train_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_label.config(text="Training failed", foreground=self.colors['error'])
        messagebox.showerror("Training Failed", f"An error occurred during training:\n\n{error_message}")
    
    def update_results_display(self, results, report_path=None):
        """Update the results display tabs"""
        # Update training history
        history = results['history'].history
        history_text = "Training History:\n" + "="*50 + "\n\n"
        
        for epoch in range(len(history['loss'])):
            history_text += f"Epoch {epoch + 1}:\n"
            history_text += f"  Loss: {history['loss'][epoch]:.4f}\n"
            history_text += f"  Accuracy: {history['accuracy'][epoch]:.4f}\n"
            history_text += f"  Val Loss: {history['val_loss'][epoch]:.4f}\n"
            history_text += f"  Val Accuracy: {history['val_accuracy'][epoch]:.4f}\n\n"
        
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(1.0, history_text)
        
        # Update model info
        model_info = f"Model Information:\n" + "="*50 + "\n\n"
        model_info += f"Model Name: {os.path.basename(results['model_path'])}\n"
        model_info += f"Label Names: {', '.join(results['label_names'])}\n"
        model_info += f"Number of Classes: {len(results['label_names'])}\n"
        model_info += f"Total Parameters: {results['model'].count_params():,}\n"
        model_info += f"Model Path: {results['model_path']}\n"
        
        if 'config_path' in results:
            model_info += f"Configuration Path: {results['config_path']}\n"
        
        self.model_info_text.delete(1.0, tk.END)
        self.model_info_text.insert(1.0, model_info)
        
        # Update files list
        self.files_listbox.delete(0, tk.END)
        files_to_show = [results['model_path']]
        
        if 'config_path' in results:
            files_to_show.append(results['config_path'])
        
        if report_path:
            files_to_show.append(report_path)
        
        # Add wave files directory
        wave_dir = os.path.join(os.getcwd(), "wave_files")
        if os.path.exists(wave_dir):
            files_to_show.append(wave_dir)
        
        for file_path in files_to_show:
            self.files_listbox.insert(tk.END, file_path)
    
    def auto_open_results(self, model_path, report_path):
        """Automatically open result files"""
        import subprocess
        import platform
        
        try:
            if report_path and os.path.exists(report_path):
                if platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', report_path])
                elif platform.system() == 'Windows':  # Windows
                    os.startfile(report_path)
                else:  # Linux
                    subprocess.call(['xdg-open', report_path])
        except Exception as e:
            print(f"Could not automatically open files: {e}")
    
    def open_selected_file(self):
        """Open the selected file in the files list"""
        selection = self.files_listbox.curselection()
        if selection:
            file_path = self.files_listbox.get(selection[0])
            self.open_file_or_folder(file_path)
    
    def open_results_folder(self):
        """Open the results folder"""
        results_dir = os.path.join(os.getcwd(), "saved_models")
        self.open_file_or_folder(results_dir)
    
    def open_file_or_folder(self, path):
        """Open a file or folder using the system default application"""
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', path])
            elif platform.system() == 'Windows':  # Windows
                if os.path.isdir(path):
                    os.startfile(path)
                else:
                    os.startfile(path)
            else:  # Linux
                subprocess.call(['xdg-open', path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open {path}:\n{e}")
    
    def save_configuration(self):
        """Save current configuration to a JSON file"""
        config = {
            'model_name': self.model_name_var.get(),
            'epochs': self.epochs_var.get(),
            'sampling_rate': self.sampling_rate_var.get(),
            'time_period': self.time_period_var.get(),
            'data_path': self.data_path_var.get(),
            'selected_folders': [folder for folder, var in self.folder_vars.items() if var.get()],
            'auto_report': self.auto_report_var.get(),
            'save_data': self.save_data_var.get(),
            'auto_open': self.auto_open_var.get(),
            'created_at': datetime.now().isoformat()
        }
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save Configuration"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save configuration:\n{e}")
    
    def load_configuration(self):
        """Load configuration from a JSON file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Configuration"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Apply configuration
                self.model_name_var.set(config.get('model_name', 'classifier_model'))
                self.epochs_var.set(config.get('epochs', '10'))
                self.sampling_rate_var.set(config.get('sampling_rate', '1600'))
                self.time_period_var.set(config.get('time_period', '10'))
                self.data_path_var.set(config.get('data_path', self.data_path))
                self.auto_report_var.set(config.get('auto_report', True))
                self.save_data_var.set(config.get('save_data', True))
                self.auto_open_var.set(config.get('auto_open', True))
                
                # Update data path and refresh folders
                self.data_path = self.data_path_var.get()
                self.refresh_folders()
                
                # Set selected folders
                selected_folders = config.get('selected_folders', [])
                for folder, var in self.folder_vars.items():
                    var.set(folder in selected_folders)
                
                self.update_folder_info()
                messagebox.showinfo("Success", f"Configuration loaded from:\n{file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load configuration:\n{e}")

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        # You can add an icon file here
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"+{x}+{y}")
    
    # Create application
    app = EnhancedModelTrainerGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.is_training:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit?"):
                app.is_training = False
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI
    root.mainloop()

if __name__ == "__main__":
    main()
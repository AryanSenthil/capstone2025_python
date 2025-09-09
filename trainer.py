import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
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

# Global parameters
sampling_rate = 1600  # Hz
train_size = 0.8
time_period = 10
test_size = 0.1
valid_size = 0.1
random_state = 42
EPOCHS = 10

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

def save_wav_files(wav_files, base_folder_path):
    # create a new version of the directory if it already exists
    version = 1
    while os.path.exists(base_folder_path):
        base_folder_path = base_folder_path.rstrip('/')+f"_v{version}"
        version += 1
    # Create a subdirectory for each unique class and save WAV files 
    # it should look like this 
    # main_directory/
    # ...class_a/
    # ......a_audio_1.wav
    # ......a_audio_2.wav
    # ...class_b/
    # ......b_audio_1.wav
    # ......b_audio_2.wav

    for audio_binary, damage_type in wav_files:
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

def wav_generator(data, sampling_rate):
    classification_data = extract_data_and_type(data)
    # Normalize the data
    normalized_data = [(normalize_data(current_sequence), damage_type) for current_sequence, damage_type in classification_data]
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

def read_csv_files(path, sampling_rate, period):
    data_for_model = []
    all_data = [] # ((time,current), type, deformation) for each file

    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith('.csv'):
                    file_path = os.path.join(root, filename)
                    all_data.append(process_csv_file(file_path))
    elif os.path.isfile(path) and path.endswith('.csv'):
        all_data.append(process_csv_file(path))
    else:
        raise ValueError(f"The path provided is neither a CSV file nor a directory: {path}")

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
        new_time = np.arange(0, max_time, interval)
        new_current = interpolation_func(new_time)

        new_time_current = list(zip(new_time, new_current))
        interpolated_data.append((new_time_current, type_of_damage))
    return interpolated_data

def generate_model_evaluation_pdf(model, history, test_spectrogram_ds, label_names, 
                                model_path=None, output_path=None, figsize=(12, 8)):
    """
    Generate a comprehensive PDF report with model evaluation metrics and plots.
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained TensorFlow/Keras model
    history : tf.keras.callbacks.History
        Training history object containing metrics
    test_spectrogram_ds : tf.data.Dataset
        Test dataset for evaluation
    label_names : list
        List of class/label names
    model_path : str, optional
        Path to the saved model (for documentation)
    output_path : str, optional
        Output path for the PDF file. If None, generates timestamp-based filename
    figsize : tuple, optional
        Figure size for plots (width, height)
    
    Returns:
    --------
    str : Path to the generated PDF file
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

def train_classification_model(data_paths, model_save_name, sampling_rate=16000, time_period=10, epochs=10):
    """
    Train a classification model from CSV data and save it.
    
    Args:
        data_paths (list): List of paths to directories containing CSV files
        model_save_name (str): Name for the saved model (without extension)
        sampling_rate (int): Audio sampling rate (default: 16000)
        time_period (float): Time period for audio generation (default: 10)
        epochs (int): Number of training epochs (default: 10)
    
    Returns:
        tuple: (model, history, save_path) - The trained model, training history, and save path
    """
    
    # Flatten data from multiple paths
    print("Loading data from paths...")
    data = []
    for path in data_paths:
        data.extend(read_csv_files(path, sampling_rate, period=time_period))
    
    # Generate wav files
    print("Generating wav files...")
    wav_files = wav_generator(data, sampling_rate)
    
    # Save wav files
    wave_folder = os.path.join(os.getcwd(), "wave_files/wave_files")
    directory = save_wav_files(wav_files, wave_folder)
    
    # Get audio properties
    print("Analyzing audio files...")
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
    print("Creating TensorFlow datasets...")
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
    print("Creating spectrogram datasets...")
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
    print("Building model...")
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
    print("Training model...")
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=epochs,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )
    
    # Test model
    print("Testing model...")
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
    print(f"Saving model as '{model_save_name}'...")
    export_model = ExportModel(model)
    save_model_directory = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_model_directory, exist_ok=True)
    model_path = os.path.join(save_model_directory, model_save_name)
    tf.saved_model.save(export_model, model_path)
    
    print(f"Model saved successfully to: {model_path}")
    
    return {
        'model': model,
        'history': history,
        'model_path': model_path,
        'test_spectrogram_ds': test_spectrogram_ds,
        'label_names': label_names,
        'train_spectrogram_ds': train_spectrogram_ds,
        'val_spectrogram_ds': val_spectrogram_ds
    }

class ModelTrainerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Classification Model Trainer")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Configure window styling
        self.root.configure(bg='#f0f0f0')
        
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
        self.setup_ui()
    
    def setup_styles(self):
        """Configure custom styles for the interface"""
        style = ttk.Style()
        
        # Configure modern button style
        style.configure('Modern.TButton', 
                       padding=(10, 8),
                       font=('Arial', 10))
        
        # Configure title label style
        style.configure('Title.TLabel',
                       font=('Arial', 18, 'bold'),
                       background='#f0f0f0',
                       foreground='#2c3e50')
        
        # Configure section header style
        style.configure('Header.TLabel',
                       font=('Arial', 12, 'bold'),
                       background='#f0f0f0',
                       foreground='#34495e')
        
        # Configure frame style with padding
        style.configure('Card.TFrame',
                       relief='solid',
                       borderwidth=1,
                       background='white')
        
    def setup_ui(self):
        # Main frame with modern styling
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.configure(style='Card.TFrame')
        
        # Title with improved styling
        title_label = ttk.Label(main_frame, text="Classification Model Trainer", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 30))
        
        # Folder selection section with card styling
        folder_section = ttk.LabelFrame(main_frame, text="Data Folders", padding="15")
        folder_section.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 25), ipady=5)
        
        folder_info = ttk.Label(folder_section, text="Select one or more folders containing CSV training data:")
        folder_info.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Create frame for checkboxes with better styling
        checkbox_frame = ttk.Frame(folder_section)
        checkbox_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Scrollable frame for checkboxes with improved appearance
        canvas = tk.Canvas(checkbox_frame, height=150, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(checkbox_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create checkboxes for each folder with better spacing
        self.folder_vars = {}
        if self.folders:
            for i, folder in enumerate(self.folders):
                var = tk.BooleanVar()
                self.folder_vars[folder] = var
                checkbox = ttk.Checkbutton(scrollable_frame, text=folder, variable=var)
                checkbox.grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
        else:
            no_folders_label = ttk.Label(scrollable_frame, text="No data folders found in the data directory")
            no_folders_label.grid(row=0, column=0, padx=10, pady=20)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add select all/none buttons
        button_frame = ttk.Frame(folder_section)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        select_all_btn = ttk.Button(button_frame, text="Select All", command=self.select_all_folders)
        select_all_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        select_none_btn = ttk.Button(button_frame, text="Select None", command=self.clear_selection)
        select_none_btn.pack(side=tk.LEFT)
        
        # Model configuration section
        config_section = ttk.LabelFrame(main_frame, text="Model Configuration", padding="15")
        config_section.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 25), ipady=5)
        
        # Model name
        model_name_label = ttk.Label(config_section, text="Model Name:")
        model_name_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.model_name_var = tk.StringVar(value="classifier_model")
        model_name_entry = ttk.Entry(config_section, textvariable=self.model_name_var, width=40, font=('Arial', 10))
        model_name_entry.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Training parameters with better layout
        params_frame = ttk.Frame(config_section)
        params_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Epochs with improved layout
        epochs_label = ttk.Label(params_frame, text="Training Epochs:")
        epochs_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        epochs_frame = ttk.Frame(params_frame)
        epochs_frame.grid(row=1, column=0, sticky=tk.W)
        
        self.epochs_var = tk.StringVar(value="10")
        epochs_entry = ttk.Entry(epochs_frame, textvariable=self.epochs_var, width=15, font=('Arial', 10))
        epochs_entry.pack(side=tk.LEFT)
        
        epochs_help = ttk.Label(epochs_frame, text="(Recommended: 10-50)", foreground="gray")
        epochs_help.pack(side=tk.LEFT, padx=(10, 0))
        
        # Options section
        options_frame = ttk.Frame(config_section)
        options_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(15, 0))
        
        # Auto-generate report checkbox with better styling
        self.auto_report_var = tk.BooleanVar(value=True)
        auto_report_check = ttk.Checkbutton(options_frame, text="Generate evaluation report (PDF)", 
                                           variable=self.auto_report_var)
        auto_report_check.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        report_help = ttk.Label(options_frame, text="Creates detailed performance analysis with charts", 
                               foreground="gray", font=('Arial', 9))
        report_help.grid(row=1, column=0, sticky=tk.W, padx=(25, 0))
        
        # Action buttons with improved styling
        action_section = ttk.Frame(main_frame)
        action_section.grid(row=4, column=0, columnspan=2, pady=(25, 20))
        
        self.train_button = ttk.Button(action_section, text="Start Training", 
                                      command=self.start_training, style='Modern.TButton')
        self.train_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Additional info label
        info_label = ttk.Label(action_section, text="Training may take several minutes depending on data size", 
                              foreground="gray", font=('Arial', 9))
        info_label.pack(side=tk.LEFT)
        
        # Status section with better visual hierarchy
        status_section = ttk.LabelFrame(main_frame, text="Status", padding="15")
        status_section.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 0))
        
        # Status label with better styling
        self.status_label = ttk.Label(status_section, text="Ready to start training", 
                                     font=('Arial', 10), foreground="#27ae60")
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Progress bar with better styling
        self.progress = ttk.Progressbar(status_section, mode='indeterminate', length=400)
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Progress percentage label (initially hidden)
        self.progress_label = ttk.Label(status_section, text="", font=('Arial', 9), foreground="gray")
        self.progress_label.grid(row=2, column=0, sticky=tk.W)
        
        # Configure grid weights for responsive layout
        main_frame.columnconfigure(1, weight=1)
        config_section.columnconfigure(1, weight=1)
        folder_section.columnconfigure(1, weight=1)
        status_section.columnconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def select_all_folders(self):
        """Select all available folders"""
        for var in self.folder_vars.values():
            var.set(True)
    
    def clear_selection(self):
        """Clear all folder selections"""
        for var in self.folder_vars.values():
            var.set(False)
    
    def validate_inputs(self):
        # Check if any folders are selected
        selected_folders = [folder for folder, var in self.folder_vars.items() if var.get()]
        if not selected_folders:
            messagebox.showerror("Validation Error", "Please select at least one data folder to continue.")
            return False
        
        # Check model name
        model_name = self.model_name_var.get().strip()
        if not model_name:
            messagebox.showerror("Validation Error", "Please enter a valid model name.")
            return False
            
        # Check for invalid characters in model name
        if not model_name.replace('_', '').replace('-', '').isalnum():
            messagebox.showerror("Validation Error", "Model name should only contain letters, numbers, hyphens, and underscores.")
            return False
        
        # Validate epochs
        try:
            epochs = int(self.epochs_var.get())
            if epochs <= 0 or epochs > 1000:
                raise ValueError
        except ValueError:
            messagebox.showerror("Validation Error", "Epochs must be a positive integer between 1 and 1000.")
            return False
        
        return True
    
    def start_training(self):
        if not self.validate_inputs():
            return
        
        # Disable train button and start progress with better feedback
        self.train_button.config(state='disabled', text="Training...")
        self.progress.start(10)  # Faster animation
        self.status_label.config(text="Initializing training process...", foreground="#3498db")
        self.progress_label.config(text="Please wait while the model trains...")
        
        # Start training in a separate thread
        training_thread = threading.Thread(target=self.train_model)
        training_thread.daemon = True
        training_thread.start()
    
    def train_model(self):
        try:
            # Get selected folders
            selected_folders = [folder for folder, var in self.folder_vars.items() if var.get()]
            data_paths = [os.path.join(self.data_path, folder) for folder in selected_folders]
            
            # Get parameters
            model_name = self.model_name_var.get().strip()
            epochs = int(self.epochs_var.get())
            auto_report = self.auto_report_var.get()
            
            # Update status with more descriptive messages
            self.root.after(0, lambda: self.status_label.config(text="Loading and preprocessing data...", foreground="#3498db"))
            self.root.after(0, lambda: self.progress_label.config(text="Reading CSV files and generating audio data..."))
            
            # Call your training function here
            results = train_classification_model(
                data_paths=data_paths,
                model_save_name=model_name,
                sampling_rate=sampling_rate,
                epochs=epochs
            )

            model = results['model']
            history = results['history']
            save_path = results['model_path']
            test_spectrogram_ds = results['test_spectrogram_ds']
            label_names = results['label_names']
            
            # Generate evaluation report if requested
            report_path = None
            if auto_report:
                self.root.after(0, lambda: self.status_label.config(text="Generating evaluation report...", foreground="#3498db"))
                self.root.after(0, lambda: self.progress_label.config(text="Creating charts and analysis..."))
                
                # Create report directory if it doesn't exist
                report_dir = os.path.join(os.getcwd(), "report")
                os.makedirs(report_dir, exist_ok=True)
                
                # Generate report path
                report_filename = os.path.splitext(os.path.basename(save_path))[0] + "_evaluation.pdf"
                report_path = os.path.join(report_dir, report_filename)
                
                # Generate the PDF report
                report_path = generate_model_evaluation_pdf(
                    model=model,
                    history=history,
                    test_spectrogram_ds=test_spectrogram_ds,
                    label_names=label_names,
                    model_path=save_path,
                    output_path=report_path
                )
            
            # Training completed successfully
            self.root.after(0, lambda: self.training_completed(save_path, report_path))
            
        except Exception as e:
            # Training failed
            self.root.after(0, lambda: self.training_failed(str(e)))
    
    def training_completed(self, save_path, report_path=None):
        self.progress.stop()
        self.train_button.config(state='normal', text="Start Training")
        self.progress_label.config(text="")
        
        if report_path:
            self.status_label.config(text="Training completed successfully! Model and report saved.", foreground="#27ae60")
            message = f"Model trained successfully!\n\nModel saved to:\n{save_path}\n\nReport saved to:\n{report_path}"
            
            # Try to open the PDF report automatically
            try:
                import subprocess
                import platform
                
                if platform.system() == 'Darwin':  # macOS
                    subprocess.call(['open', report_path])
                elif platform.system() == 'Windows':  # Windows
                    os.startfile(report_path)
                else:  # Linux
                    subprocess.call(['xdg-open', report_path])
            except Exception as e:
                print(f"Could not automatically open PDF: {e}")
                
        else:
            self.status_label.config(text="Training completed successfully! Model saved.", foreground="#27ae60")
            message = f"Model trained successfully!\n\nModel saved to:\n{save_path}"
        
        messagebox.showinfo("Success", message)
    
    def training_failed(self, error_message):
        self.progress.stop()
        self.train_button.config(state='normal', text="Start Training")
        self.progress_label.config(text="")
        self.status_label.config(text="Training failed - please check your data and try again", foreground="#e74c3c")
        messagebox.showerror("Training Failed", f"An error occurred during training:\n\n{error_message}\n\nPlease check your data files and try again.")

def main():
    root = tk.Tk()
    app = ModelTrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
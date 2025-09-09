import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import numpy as np
import tensorflow as tf
from keras import layers, models
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from keras import layers 
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models


sampling_rate = 1600  #Hz
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
#%%


def train_classification_model(data_paths, model_save_name, sampling_rate=16000, time_period=10, epochs=10):
    """
    Train an classification model from CSV data and save it.
    
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
        self.root.title("classification Model Trainer")
        self.root.geometry("2000x1000")
        
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
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="classification Model Trainer", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Folder selection section
        folder_label = ttk.Label(main_frame, text="Select Data Folders:", font=("Arial", 12, "bold"))
        folder_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Create frame for checkboxes
        checkbox_frame = ttk.Frame(main_frame)
        checkbox_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Scrollable frame for checkboxes
        canvas = tk.Canvas(checkbox_frame, height=200)
        scrollbar = ttk.Scrollbar(checkbox_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create checkboxes for each folder
        self.folder_vars = {}
        for i, folder in enumerate(self.folders):
            var = tk.BooleanVar()
            self.folder_vars[folder] = var
            checkbox = ttk.Checkbutton(scrollable_frame, text=folder, variable=var)
            checkbox.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Model name section
        model_name_label = ttk.Label(main_frame, text="Model Name:", font=("Arial", 12, "bold"))
        model_name_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        
        self.model_name_var = tk.StringVar(value="classifier_model")
        model_name_entry = ttk.Entry(main_frame, textvariable=self.model_name_var, width=30)
        model_name_entry.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Parameters section
        params_label = ttk.Label(main_frame, text="Training Parameters:", font=("Arial", 12, "bold"))
        params_label.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        # Epochs
        epochs_label = ttk.Label(main_frame, text="Epochs:")
        epochs_label.grid(row=6, column=0, sticky=tk.W)
        
        self.epochs_var = tk.StringVar(value="10")
        epochs_entry = ttk.Entry(main_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=6, column=1, sticky=tk.W, padx=(10, 0))
        
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=(20, 0))
        
        self.train_button = ttk.Button(button_frame, text="Train Model", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_button = ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection)
        clear_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready to train", foreground="green")
        self.status_label.grid(row=9, column=0, columnspan=2, pady=(10, 0))
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
    def clear_selection(self):
        for var in self.folder_vars.values():
            var.set(False)
    
    def validate_inputs(self):
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
        
        
        return True
    
    def start_training(self):
        if not self.validate_inputs():
            return
        
        # Disable train button and start progress
        self.train_button.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Training in progress...", foreground="blue")
        
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
            # Use the constant sampling rate defined at the top of the file
            
            # Update status
            self.root.after(0, lambda: self.status_label.config(text="Loading data...", foreground="blue"))
            
            # Call your training function here
            results = train_classification_model(
                data_paths=data_paths,
                model_save_name=model_name,
                sampling_rate=sampling_rate,
                epochs=epochs
            )

            model = results['model']
            hisory = results['history']
            save_path = results['model_path']
            
            # Training completed successfully
            self.root.after(0, lambda: self.training_completed(save_path))
            
        except Exception as e:
            # Training failed
            self.root.after(0, lambda: self.training_failed(str(e)))
    
    def training_completed(self, save_path):
        self.progress.stop()
        self.train_button.config(state='normal')
        self.status_label.config(text=f"Training completed! Model saved to: {save_path}", foreground="green")
        messagebox.showinfo("Success", f"Model trained successfully!\nSaved to: {save_path}")
    
    def training_failed(self, error_message):
        self.progress.stop()
        self.train_button.config(state='normal')
        self.status_label.config(text="Training failed", foreground="red")
        messagebox.showerror("Training Failed", f"An error occurred during training:\n{error_message}")

def main():
    root = tk.Tk()
    app = ModelTrainerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
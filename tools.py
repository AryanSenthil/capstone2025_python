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
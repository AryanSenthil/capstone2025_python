import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf
import threading
import time
from datetime import datetime
import logging
from pathlib import Path

# Configuration constants
interpolation_interval = 0.120  # s
block_interval = 7  # s
padding_time = 1.56  # s

class CSVProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Processor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Configure style
        self.setup_styles()
        
        # Variables
        self.input_path_var = tk.StringVar()
        self.damage_label_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        
        # Progress tracking
        self.total_files = 0
        self.processed_files = 0
        self.is_processing = False
        
        # Setup logging
        self.setup_logging()
        
        self.setup_ui()

    def setup_styles(self):
        """Configure custom styles for ttk widgets"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.style.configure('Title.TLabel', 
                           font=('Arial', 16, 'bold'),
                           foreground='#2c3e50',
                           background='#f0f0f0')
        
        self.style.configure('Heading.TLabel', 
                           font=('Arial', 12, 'bold'),
                           foreground='#34495e',
                           background='#f0f0f0')
        
        self.style.configure('Success.TButton',
                           background='#27ae60',
                           foreground='white',
                           font=('Arial', 11, 'bold'))
        
        self.style.map('Success.TButton',
                      background=[('active', '#2ecc71')])
        
        self.style.configure('Danger.TButton',
                           background='#e74c3c',
                           foreground='white',
                           font=('Arial', 11))
        
        self.style.map('Danger.TButton',
                      background=[('active', '#c0392b')])

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"csv_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """Setup the user interface"""
        # Main container with padding
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_container, text="CSV Processor", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Input section
        self.create_input_section(main_container)
        
        # Progress section
        self.create_progress_section(main_container)
        
        # Control buttons
        self.create_control_section(main_container)
        
        # Log section
        self.create_log_section(main_container)
        
        # Status bar
        self.create_status_bar()

    def create_input_section(self, parent):
        """Create input fields section"""
        input_frame = ttk.LabelFrame(parent, text="Input Configuration", padding=15)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Input Directory
        ttk.Label(input_frame, text="Input Directory:", style='Heading.TLabel').grid(
            row=0, column=0, sticky=tk.W, pady=5)
        
        dir_frame = ttk.Frame(input_frame)
        dir_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        dir_frame.columnconfigure(0, weight=1)
        
        self.input_entry = ttk.Entry(dir_frame, textvariable=self.input_path_var, font=('Arial', 10))
        self.input_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        browse_btn = ttk.Button(dir_frame, text="Browse", command=self.browse_input_directory)
        browse_btn.grid(row=0, column=1)
        
        # Damage Label
        ttk.Label(input_frame, text="Damage Label:", style='Heading.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=5)
        
        self.damage_entry = ttk.Entry(input_frame, textvariable=self.damage_label_var, 
                                    font=('Arial', 10))
        self.damage_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Output Folder
        ttk.Label(input_frame, text="Output Folder Name:", style='Heading.TLabel').grid(
            row=2, column=0, sticky=tk.W, pady=5)
        
        self.output_entry = ttk.Entry(input_frame, textvariable=self.output_folder_var,
                                    font=('Arial', 10))
        self.output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Configure grid weights
        input_frame.columnconfigure(1, weight=1)

    def create_progress_section(self, parent):
        """Create progress tracking section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=15)
        progress_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Overall progress
        ttk.Label(progress_frame, text="Overall Progress:", style='Heading.TLabel').pack(anchor=tk.W)
        
        self.overall_progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.overall_progress.pack(fill=tk.X, pady=5)
        
        self.overall_progress_label = ttk.Label(progress_frame, text="Ready to process")
        self.overall_progress_label.pack(anchor=tk.W, pady=(0, 10))
        
        # File progress
        ttk.Label(progress_frame, text="Current File:", style='Heading.TLabel').pack(anchor=tk.W)
        
        self.file_progress = ttk.Progressbar(progress_frame, mode='indeterminate', length=400)
        self.file_progress.pack(fill=tk.X, pady=5)
        
        self.file_progress_label = ttk.Label(progress_frame, text="No file being processed")
        self.file_progress_label.pack(anchor=tk.W)

    def create_control_section(self, parent):
        """Create control buttons section"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=15)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(anchor=tk.CENTER)
        
        self.process_btn = ttk.Button(button_frame, text="Process CSV Files", 
                                    style='Success.TButton', command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=10)
        
        self.cancel_btn = ttk.Button(button_frame, text="Cancel", 
                                   style='Danger.TButton', command=self.cancel_processing,
                                   state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=10)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear Logs", 
                                  command=self.clear_logs)
        self.clear_btn.pack(side=tk.LEFT, padx=10)

    def create_log_section(self, parent):
        """Create logging section"""
        log_frame = ttk.LabelFrame(parent, text="Processing Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for different log levels
        self.log_text.tag_configure('INFO', foreground='#2c3e50')
        self.log_text.tag_configure('SUCCESS', foreground='#27ae60', font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure('ERROR', foreground='#e74c3c', font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure('WARNING', foreground='#f39c12', font=('Consolas', 9, 'bold'))

    def create_status_bar(self):
        """Create status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def log_message(self, message, level='INFO'):
        """Add message to log with timestamp and color coding"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, formatted_message, level)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
        # Also log to file
        if level == 'ERROR':
            self.logger.error(message)
        elif level == 'WARNING':
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def clear_logs(self):
        """Clear the log text area"""
        self.log_text.delete(1.0, tk.END)

    def browse_input_directory(self):
        """Browse for input directory"""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_path_var.set(directory)
            self.log_message(f"Selected input directory: {directory}")
            
            # Count CSV files for progress tracking
            try:
                csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
                self.total_files = len(csv_files)
                self.log_message(f"Found {self.total_files} CSV files")
                self.status_var.set(f"Found {self.total_files} CSV files in directory")
            except Exception as e:
                self.log_message(f"Error reading directory: {str(e)}", 'ERROR')

    def validate_inputs(self):
        """Validate user inputs"""
        if not self.input_path_var.get():
            messagebox.showerror("Error", "Please select an input directory")
            return False
            
        if not os.path.isdir(self.input_path_var.get()):
            messagebox.showerror("Error", "Invalid input directory")
            return False
            
        if not self.damage_label_var.get():
            messagebox.showerror("Error", "Please enter a damage label")
            return False
            
        if not self.output_folder_var.get():
            messagebox.showerror("Error", "Please enter an output folder name")
            return False
            
        return True

    def start_processing(self):
        """Start processing in a separate thread"""
        if not self.validate_inputs():
            return
            
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing is already in progress")
            return
            
        # Reset progress
        self.processed_files = 0
        self.overall_progress['value'] = 0
        
        # Update UI state
        self.is_processing = True
        self.process_btn['state'] = tk.DISABLED
        self.cancel_btn['state'] = tk.NORMAL
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_files_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def cancel_processing(self):
        """Cancel processing"""
        self.is_processing = False
        self.log_message("Processing cancelled by user", 'WARNING')
        self.reset_ui_state()

    def reset_ui_state(self):
        """Reset UI to ready state"""
        self.process_btn['state'] = tk.NORMAL
        self.cancel_btn['state'] = tk.DISABLED
        self.file_progress.stop()
        self.status_var.set("Ready")
        self.file_progress_label.config(text="No file being processed")

    def process_files_thread(self):
        """Process files in separate thread"""
        try:
            input_path = self.input_path_var.get()
            damage_label = self.damage_label_var.get()
            output_folder_name = self.output_folder_var.get()
            
            base_dir = os.getcwd()
            output_folder = os.path.join(base_dir, "data", output_folder_name)
            os.makedirs(output_folder, exist_ok=True)
            
            self.log_message("Starting CSV processing...", 'INFO')
            self.log_message(f"Input directory: {input_path}")
            self.log_message(f"Output directory: {output_folder}")
            self.log_message(f"Damage label: {damage_label}")
            
            start_time = time.time()
            
            success_count = 0
            error_count = 0
            
            # Process directory
            if os.path.isdir(input_path):
                csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
                self.total_files = len(csv_files)
                
                for i, filename in enumerate(csv_files):
                    if not self.is_processing:  # Check for cancellation
                        break
                        
                    filepath = os.path.join(input_path, filename)
                    
                    # Update current file progress
                    self.root.after(0, lambda f=filename: self.update_file_progress(f))
                    
                    try:
                        self.log_message(f"Processing file: {filename}")
                        processed_df = process_csv_file(filepath)
                        write_csv(processed_df, damage_label, output_folder)
                        success_count += 1
                        self.log_message(f"Successfully processed: {filename}", 'SUCCESS')
                        
                    except Exception as e:
                        error_count += 1
                        error_msg = f"Error processing {filename}: {str(e)}"
                        self.log_message(error_msg, 'ERROR')
                        
                    # Update overall progress
                    self.processed_files += 1
                    progress_percentage = (self.processed_files / self.total_files) * 100
                    self.root.after(0, lambda p=progress_percentage: self.update_overall_progress(p))
            
            # Processing complete
            end_time = time.time()
            duration = end_time - start_time
            
            if self.is_processing:  # Only show completion if not cancelled
                self.log_message("=" * 50, 'SUCCESS')
                self.log_message(f"Processing completed!", 'SUCCESS')
                self.log_message(f"Total files: {self.total_files}")
                self.log_message(f"Successfully processed: {success_count}", 'SUCCESS')
                if error_count > 0:
                    self.log_message(f"Errors encountered: {error_count}", 'ERROR')
                self.log_message(f"Processing time: {duration:.2f} seconds")
                self.log_message("=" * 50, 'SUCCESS')
                
                self.root.after(0, lambda: messagebox.showinfo("Complete", 
                    f"Processing completed!\n\nProcessed: {success_count} files\nErrors: {error_count} files\nTime: {duration:.2f}s"))
                
        except Exception as e:
            error_msg = f"Fatal error during processing: {str(e)}"
            self.log_message(error_msg, 'ERROR')
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            
        finally:
            self.is_processing = False
            self.root.after(0, self.reset_ui_state)

    def update_overall_progress(self, percentage):
        """Update overall progress bar"""
        self.overall_progress['value'] = percentage
        self.overall_progress_label.config(
            text=f"Processed {self.processed_files}/{self.total_files} files ({percentage:.1f}%)")
        self.status_var.set(f"Processing... {self.processed_files}/{self.total_files} files")

    def update_file_progress(self, filename):
        """Update current file progress"""
        self.file_progress.start(10)  # Start animation
        self.file_progress_label.config(text=f"Processing: {filename}")


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


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVProcessorGUI(root)
    root.mainloop()
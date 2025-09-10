# main.py
import tkinter as tk
from tkinter import ttk

# Import your three apps (update module names/paths as needed)
from test_model import AudioClassifierGUI
from csv_processor import CSVProcessorGUI
from trainer import ModelTrainerGUI

def main():
    root = tk.Tk()
    root.title("All Tools")

    # Optional: theme + sizing
    style = ttk.Style()
    for candidate in ("clam", "alt", "default", "classic"):
        if candidate in style.theme_names():
            style.theme_use(candidate)
            break

    # Set a reasonable window size and center it
    w, h = 1100, 750
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    x, y = (sw - w) // 2, (sh - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.minsize(900, 600)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Create tabs and mount each app into its tab
    tab1 = ttk.Frame(notebook)
    tab2 = ttk.Frame(notebook)
    tab3 = ttk.Frame(notebook)

    notebook.add(tab1, text="Audio Classifier")
    notebook.add(tab2, text="CSV Processor")
    notebook.add(tab3, text="Model Trainer")

    # If your classes are Frame subclasses, just instantiate them with the tab as parent
    audio_app = AudioClassifierGUI(tab1)
    audio_app.pack(fill="both", expand=True)

    csv_app = CSVProcessorGUI(tab2)
    csv_app.pack(fill="both", expand=True)

    model_app = ModelTrainerGUI(tab3)
    model_app.pack(fill="both", expand=True)

    root.mainloop()

if __name__ == "__main__":
    main()

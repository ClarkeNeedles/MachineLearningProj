import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os

def read_and_clean_csv(filepath):
    """
    Reads a CSV file using the correct separator
    Uses tab if the filename contains "JumpingPhoneInHand" or "WalkingPhoneInBackPocket",
    otherwise uses comma. This is because the data for those two .csv files were exported
    as a different type than the other .csv files. Also cleans up the column names
    """

    # Setting the seperator
    if "JumpingPhoneInHand" in filepath or "WalkingPhoneInBackPocket" in filepath:
        sep = "\t"
    else:
        sep = ","

    df = pd.read_csv(filepath, sep=sep)

    # For the case that the data was misparsed
    # As in the data is read as a singular column
    # Because of the way the data was exported
    if df.shape[1] == 1:
        new_cols = df.columns[0].split(sep) # Splitting the first row (headers) based on the seperator
        df = df[df.columns[0]].str.split(sep, expand=True) # Split the data into the proper columns
        df.columns = new_cols # Assign the separated headers to the corresponding column

    # Getting rid of extra white space before returning
    df.columns = df.columns.str.replace('"', '').str.strip()
    print(f"Read file {os.path.basename(filepath)} with shape: {df.shape}")
    return df


def preprocess_data(df):
    """
    Preprocesses the DataFrame:
      - Renames columns,
      - Converts values to numeric,
      - Fills missing values (linear interpolation then forward fill),
      - Applies a moving average filter with window=6 (to retain more details)
    """

    df = df.copy()
    # Renaming the columns for simplicity
    if 'Time (s)' in df.columns:
        df.rename(columns={
            'Time (s)': 'time',
            'Linear Acceleration x (m/s^2)': 'x',
            'Linear Acceleration y (m/s^2)': 'y',
            'Linear Acceleration z (m/s^2)': 'z',
            'Absolute acceleration (m/s^2)': 'abs_acc'
        }, inplace=True)

    # Change string value to numeric values
    for col in ['time', 'x', 'y', 'z', 'abs_acc']:
        df[col] = pd.to_numeric(df[col])

    # Linear interpolation and forward fill
    df.interpolate(method='linear', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Apply moving average filter (window=6) to preserve more detail
    df['x_filtered'] = df['x'].rolling(window=6, center=True).mean()
    df['y_filtered'] = df['y'].rolling(window=6, center=True).mean()
    df['z_filtered'] = df['z'].rolling(window=6, center=True).mean()

    print("After preprocessing, sample data:")
    print(df.head())
    return df


def segment_data_by_time(pp_df, segment_duration=5):
    """
    Segments preprocessed data into windows of approximately 'segment_duration' seconds using the time column
    Returns a list of segments (each segment is a 2D array with columns: x, y, z)
    """

    # Dropping any extra NaN values and converting to numpy arrays
    time_array = pp_df['time'].dropna().to_numpy()
    x_array = pp_df['x_filtered'].dropna().to_numpy()
    y_array = pp_df['y_filtered'].dropna().to_numpy()
    z_array = pp_df['z_filtered'].dropna().to_numpy()

    segments = []
    start_idx = 0
    while start_idx < len(time_array):
        start_time = time_array[start_idx]
        # Using numpy search method
        end_idx = np.searchsorted(time_array, start_time + segment_duration, side='left')
        # If the segment has more than 1 data point, it creates the 2D array segment
        if end_idx - start_idx > 1:
            seg = np.column_stack((x_array[start_idx:end_idx],
                                   y_array[start_idx:end_idx],
                                   z_array[start_idx:end_idx]))
            # Append the created segment to the segments list
            segments.append(seg)
        # Update the start index for the next segment section
        start_idx = end_idx

    print(f"Segmented into {len(segments)} segments of ~{segment_duration} seconds each.")
    return segments


def extract_features(segment):
    """
    Extracts 14 features from a segment
    For each axis (x, y, z), computes:
      - Mean, Standard Deviation, Minimum, Maximum
    Additionally, computes:
      - max_jerk: the maximum absolute difference between consecutive samples across all axes
      - max_acc: the maximum L2 norm (overall acceleration magnitude) across the segment
    Returns a dictionary of 14 features
    """

    features = {}
    axes = ['x', 'y', 'z']
    # For each axis, compute basic statistics
    for axis in axes:
        data = segment[:, axes.index(axis)]
        features[f'mean_{axis}'] = np.mean(data)
        features[f'std_{axis}'] = np.std(data)
        features[f'min_{axis}'] = np.min(data)
        features[f'max_{axis}'] = np.max(data)

    # Compute max jerk for each axis, then take the maximum
    jerk_vals = []
    # for each column/axis
    for i in range(segment.shape[1]):
        axis_data = segment[:, i]
        jerk = np.abs(np.diff(axis_data))
        if len(jerk) > 0:
            jerk_vals.append(np.max(jerk))
    features['max_jerk'] = np.max(jerk_vals) if jerk_vals else 0

    # Compute max acceleration as the maximum L2 norm across all samples
    norms = np.linalg.norm(segment, axis=1)
    features['max_acc'] = np.max(norms)

    return features


def get_label_from_filename(filepath):
    """
    Returns 1 if "Jumping" is in the file name, 0 if "Walking" is in the file name
    """

    base = os.path.basename(filepath)

    if "Jumping" in base:
        return 1
    elif "Walking" in base:
        return 0

"""
File Information and HDF5 Storage
"""
# List of tuples for what file is assigned to which group member
file_info = [
    ('../Data/JumpingPhoneInBackPocket.csv', 'Ben M'),
    ('../Data/WalkingPhoneInBackPocket.csv', 'Ben M'),
    ('../Data/JumpingPhoneInFrontPocket.csv', 'Clarke N'),
    ('../Data/WalkingPhoneInFrontPocket.csv', 'Clarke N'),
    ('../Data/JumpingPhoneInHand.csv', 'Andrew P'),
    ('../Data/WalkingPhoneInHand.csv', 'Andrew P')
]

#Read and preprocess each file
#Store in dictionaries
raw_data_dict = {}
pp_data_dict = {}
for filepath, user in file_info:
    raw_data_dict[filepath] = read_and_clean_csv(filepath)
    pp_data_dict[filepath] = preprocess_data(raw_data_dict[filepath])

# Create HDF5 file structure
with h5py.File('project_data.h5', 'w') as h5:
    raw_group = h5.create_group('Raw Data')
    pp_group = h5.create_group('Pre-processed Data')
    segmented_group = h5.create_group('Segmented Data')
    # Create subgroups for each member
    for user in ['Andrew P', 'Ben M', 'Clarke N']:
        raw_group.create_group(user)
        pp_group.create_group(user)
    # Create groups for segmented data
    segmented_group.create_group('Train')
    segmented_group.create_group('Test')
    # Store raw and preprocessed data using file names as dataset names
    for filepath, user in file_info:
        ds_name = os.path.basename(filepath)
        raw_group[user].create_dataset(ds_name, data=raw_data_dict[filepath].to_numpy())
        pp_group[user].create_dataset(ds_name, data=pp_data_dict[filepath].to_numpy())
print("Data stored in HDF5 file.")

"""
Print HDF5 File Structure
"""
with h5py.File('project_data.h5', 'r') as h5:
    print("HDF5 File Structure:")
    for grp in h5.keys():
        print(f"\nGroup: {grp}")
        for subgrp in h5[grp].keys():
            print(f"  Subgroup: {subgrp}")
            for ds in h5[grp][subgrp].keys():
                print(f"    Dataset: {ds}")

"""
Data Visualizations
"""
# Visualize preprocessed data from JumpingPhoneInHand.csv
pp_demo = pp_data_dict['../Data/JumpingPhoneInHand.csv']
plt.figure(figsize=(10, 4))
plt.plot(pp_demo['time'], pp_demo['x'], label='Original x')
plt.plot(pp_demo['time'], pp_demo['x_filtered'], label='Filtered x', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Preprocessed JumpingPhoneInHand Data')
plt.legend()
plt.show()

# Visualize an example segment using time-based segmentation
segments_example = segment_data_by_time(pp_demo, segment_duration=5)
print("Example segments count (time-based):", len(segments_example))
if segments_example:
    seg0 = segments_example[0]
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(seg0)), seg0[:, 0], label='x')
    plt.plot(np.arange(len(seg0)), seg0[:, 1], label='y')
    plt.plot(np.arange(len(seg0)), seg0[:, 2], label='z')
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration")
    plt.title("Example 5-Second Segment from JumpingPhoneInHand")
    plt.legend()
    plt.show()

"""
Combine Segments and Train Classifier
"""
all_segments = []
all_labels = []
# For each file, segment the preprocessed data into 5-second segments
for filepath, user in file_info:
    pp_df = pp_data_dict[filepath]
    label = get_label_from_filename(filepath)
    segments_list = segment_data_by_time(pp_df, segment_duration=5)
    if len(segments_list) == 0:
        continue
    # Randomize the segments from this file
    np.random.shuffle(segments_list)
    all_segments.extend(segments_list)
    all_labels.extend(np.full(len(segments_list), label))
# Variable-length segments stored as an object array
all_segments = np.array(all_segments, dtype=object)
all_labels = np.array(all_labels)
print("Total number of segments combined:", len(all_segments))

# Extract features from each segment
features_all = [extract_features(seg) for seg in all_segments]
features_df = pd.DataFrame(features_all)
print("Combined Features DataFrame (first 5 rows):")
print(features_df.head())

# Split features (X) and labels (y) into training (90%) and testing (10%) sets
X_train_feat, X_test_feat, y_train, y_test = train_test_split(features_df, all_labels, test_size=0.1, random_state=42,
                                                              shuffle=True)
print("Training feature set shape:", X_train_feat.shape)
print("Testing feature set shape:", X_test_feat.shape)

# Normalize features
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_feat)
X_test_norm = scaler.transform(X_test_feat)

# Train a Logistic Regression classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_norm, y_train)
print("Classifier trained.")

# Evaluate the model
preds = model.predict(X_test_norm)
acc = accuracy_score(y_test, preds)
print("Final Model Test Accuracy:", acc)
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print("Classification Report:")
print(classification_report(y_test, preds))


"""
Prediction Pipeline for New Input
"""
def predict_from_file(input_filepath, segment_duration=5):
    """
    Processes a new CSV file:
      - Reads and preprocesses the data,
      - Segments it by time into 5-second windows,
      - Extracts features,
      - Normalizes the features using the trained scaler,
      - Predicts labels using the trained classifier
    Returns a DataFrame with segment indices and predicted labels
    """

    df = read_and_clean_csv(input_filepath)
    pp_df = preprocess_data(df)
    segments_list = segment_data_by_time(pp_df, segment_duration=segment_duration)
    if len(segments_list) == 0:
        raise ValueError("Not enough data for segmentation.")

    features = [extract_features(seg) for seg in segments_list]
    features_df = pd.DataFrame(features)
    features_norm = scaler.transform(features_df)
    predictions = model.predict(features_norm)
    output_df = pd.DataFrame({
        'Segment_Index': np.arange(len(predictions)),
        'Predicted_Label': predictions
    })

    print("Prediction complete. First 5 predictions:")
    print(output_df.head())
    return output_df, segments_list


"""
Desktop App using Tkinter
"""
def run_app():
    root = tk.Tk()
    root.title("Activity Classification App")
    # App size, ensure fixed size
    root.geometry("650x375")
    root.resizable(False, False)

    # Use a ttk style object
    style = ttk.Style()
    style.theme_use('clam')

    # Colours
    bg_color = '#f5f5f5'
    button_color = '#4a7a8c'
    text_color = '#333333'

    # Assigning background colour
    root.configure(bg=bg_color)

    selected_file = tk.StringVar()

    # Main container
    main_frame = ttk.Frame(root, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Header
    header = ttk.Label(main_frame,
                       text="Activity Classification App",
                       font=('Helvetica', 16, 'bold'),
                       foreground=button_color)
    header.pack(pady=(0, 15))

    # File section
    file_frame = ttk.LabelFrame(main_frame, text="1. Load Data", padding=10)
    file_frame.pack(fill=tk.X, pady=5)

    def load_file():
        # Only allow .csv files
        file_path = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV files", "*.csv")]
        )
        # If a file path is chosen
        if file_path:
            selected_file.set(file_path)
            # Update the load csv file label
            lbl_file.config(
                text="Loaded: " + os.path.basename(file_path),
                # Green
                foreground='#2E7D32'
            )
            # Enable run button
            btn_run.state(['!disabled'])
            print(f"File loaded: {file_path}")

    # Load csv file button
    btn_load = ttk.Button(
        file_frame,
        text="Browse CSV File",
        command=load_file,
        style='Accent.TButton'
    )
    btn_load.pack(pady=5, fill=tk.X)

    # Load csv file status label
    lbl_file = ttk.Label(
        file_frame,
        text="No file selected",
        # Gray
        foreground='#616161',
        font=('Helvetica', 9)
    )
    lbl_file.pack(pady=5)

    # Prediction section
    pred_frame = ttk.LabelFrame(main_frame, text="2. Run Analysis", padding=10)
    pred_frame.pack(fill=tk.X, pady=10)

    # For saving the .csv file
    def save_output(output_df):
        out_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if out_path:
            output_df.to_csv(out_path, index=False)
            messagebox.showinfo(
                "Success",
                f"Results saved to:\n{os.path.basename(out_path)}",
                parent=root
            )
            print(f"Output saved to: {out_path}")

    def run_prediction():
        # Error message
        if not selected_file.get():
            messagebox.showerror(
                "Error",
                "Please select a CSV file first",
                parent=root
            )
            return

        # Running prediction
        try:
            output_df, segs = predict_from_file(selected_file.get(), segment_duration=5)
            save_output(output_df)

            # Configure plot
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(9, 5))

            # Define a discrete colormap: blue for 0 (walking), red for 1 (jumping)
            cmap = ListedColormap(['#1f77b4', '#d62728'])

            scatter = ax.scatter(
                output_df['Segment_Index'],
                output_df['Predicted_Label'],
                c=output_df['Predicted_Label'],
                cmap=cmap,
                marker='o',
                s=120,
                alpha=0.8,
                edgecolors='w',
                linewidth=0.5
            )

            ax.set_xlabel("Segment Index", fontsize=11)
            ax.set_ylabel("Activity Class", fontsize=11)
            ax.set_title("Prediction Results (5-Second Segments)", fontsize=13, pad=15)
            ax.grid(True, linestyle=':', alpha=0.7)

            # Optional: Create custom colorbar with labels for discrete values
            cbar = plt.colorbar(scatter, ticks=[0, 1])
            cbar.ax.set_yticklabels(['Walking', 'Jumping'])
            cbar.set_label('Activity Type', rotation=270, labelpad=15)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror(
                "Processing Error",
                f"Could not process file:\n{str(e)}",
                parent=root
            )
            print("Error:", e)

    # Run prediction button
    btn_run = ttk.Button(
        pred_frame,
        text="Run Activity Prediction",
        command=run_prediction,
        style='Accent.TButton',
        # Disabled until file selected
        state='disabled'
    )
    btn_run.pack(pady=0, fill=tk.X)

    # Style configurations
    style.configure('TLabelFrame', background=bg_color, bordercolor='#e0e0e0')
    style.configure('TLabelFrame.Label', font=('Helvetica', 10, 'bold'))
    style.configure('Accent.TButton',
                    foreground='white',
                    background=button_color,
                    font=('Helvetica', 10),
                    padding=5)
    style.map('Accent.TButton',
              background=[('active', '#3a6a7c'), ('pressed', '#2a5a6c')])

    root.mainloop()


if __name__ == "__main__":
    run_app()

import h5py
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm

# Paths to your H5 files
h5_file_paths = [
    "D:/DL_DATASET/processed_images_zip1.h5"
]

# Path to your metadata CSV
metadata_csv_path = r"C:\Shivangi\college\Sem 5\Deep Learning\DL project\greymatter\20_subjects_small_model\test_merged_mri_metadata.csv"

# Load metadata
metadata_df = pd.read_csv(metadata_csv_path)

# Ensure that the total number of images matches across H5 files and metadata
total_images = sum([h5py.File(h5, 'r')['images'].shape[0] for h5 in h5_file_paths])
assert total_images == len(metadata_df), "Mismatch between total images in H5 files and metadata entries."

# Initialize the patient-wise index dictionary
patient_index = {}

# Initialize a variable to keep track of the current image index across H5 files
current_global_idx = 0

# Iterate over each H5 file and map images to patient IDs
for h5_file in h5_file_paths:
    with h5py.File(h5_file, 'r') as hdf5_file:
        num_images = hdf5_file['images'].shape[0]
        
        for local_idx in tqdm(range(num_images), desc=f"Indexing {h5_file}"):
            # Get the corresponding row in metadata_df
            row = metadata_df.iloc[current_global_idx]
            patient_id = row['patient_id']
            
            # Initialize the list for the patient if not already
            if patient_id not in patient_index:
                patient_index[patient_id] = []
            
            # Append a tuple of (H5 file path, image index within the H5 file)
            patient_index[patient_id].append((h5_file, local_idx))
            
            current_global_idx += 1

# Save the patient-wise index to a JSON file
index_output_path = "D:/DL_DATASET/patient_wise_index.json"
with open(index_output_path, 'w') as f:
    json.dump(patient_index, f)

print(f"Patient-wise index created and saved to {index_output_path}.")

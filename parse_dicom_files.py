import os
import re
from datetime import datetime
import pandas as pd
import pydicom
from tqdm import tqdm

# ############################### Configuration ###################################
'''
#########################################################################
##################### Change these file paths ##############################
#########################################################################
'''

# Base directory containing DICOM files
base_dir = "/Users/Agaaz/Downloads/ADNI_nimrat"  # Adjust this to your actual directory

# Output metadata CSV path
output_path = "/Users/Agaaz/Downloads/zip10_metadata_from_dcm1.csv"

# ############################### Functions ########################################

def parse_dicom_filename(file_name):
    """
    Parses the DICOM filename to extract patient_id, scan_type, and datetime.

    Args:
        file_name (str): The name of the DICOM file.

    Returns:
        dict: A dictionary containing parsed metadata.
    """
    # Regex patterns
    patient_id_pattern = r"(\d{3}_S_\d{4})"  # Matches patient ID format like 003_S_6644
    datetime_pattern = r"(\d{8})(\d{6})"     # Matches datetime in format YYYYMMDDHHMMSS

    # Extract patient ID using regex
    patient_id_match = re.search(patient_id_pattern, file_name)
    patient_id = patient_id_match.group(1) if patient_id_match else None

    # Extract scan type from the filename (assumes it is the 4th part if available)
    parts = file_name.split('_')
    scan_type = parts[3] if len(parts) >= 4 else None

    # Extract datetime using regex
    datetime_match = re.search(datetime_pattern, file_name)
    if datetime_match:
        date_part = datetime_match.group(1)  # YYYYMMDD
        time_part = datetime_match.group(2)  # HHMMSS
        datetime_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        try:
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            datetime_obj = None
    else:
        datetime_obj = None

    # Return the parsed data as a dictionary
    return {
        'patient_id': patient_id,
        'scan_type': scan_type,
        'datetime': datetime_obj,
        'filename': file_name
    }

def process_folders(base_dir):
    """
    Traverses the base directory to locate DICOM files, parses filenames,
    determines the number of slices per file, and creates metadata entries
    for each slice.

    Args:
        base_dir (str): The base directory containing DICOM files.

    Returns:
        pd.DataFrame: A DataFrame containing metadata for all image slices.
    """
    # List to store parsed data
    all_data = []

    # Traverse directories with os.walk()
    for root, dirs, files in os.walk(base_dir):
        if 'I' in os.path.basename(root):  # Only look at folders containing DICOM files
            # Loop through files and try parsing the DICOM filenames
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    # Parse the filename
                    parsed_data = parse_dicom_filename(file_name)
                    if parsed_data and parsed_data['patient_id']:  # Ensure patient_id exists
                        # Now, read the DICOM file to get the number of frames (slices)
                        try:
                            dicom_data = pydicom.dcmread(file_path)
                            img = dicom_data.pixel_array
                            if img.ndim == 3:
                                num_slices = img.shape[0]
                            elif img.ndim == 2:
                                num_slices = 1
                            else:
                                raise ValueError(f"Unsupported image dimensions: {img.shape}")
                        except Exception as e:
                            print(f"Error reading DICOM file {file_path}: {e}")
                            # Optionally, skip or add a default number of slices
                            num_slices = 0  # Or handle differently

                        if num_slices > 0:
                            for slice_num in range(num_slices):
                                # Create a new entry for each slice
                                entry = {
                                    'patient_id': parsed_data['patient_id'],
                                    'scan_type': parsed_data['scan_type'],
                                    'datetime': parsed_data['datetime'],
                                    'filename': parsed_data['filename'],
                                    'file_path': file_path,
                                    'slice_number': slice_num + 1  # 1-based indexing
                                }
                                all_data.append(entry)
                        else:
                            # If no slices found, decide how to handle
                            # For example, skip or add a placeholder entry
                            print(f"No slices found in DICOM file {file_path}. Skipping.")
                            # Optionally, add an entry with slice_number as 0 or None
                            # entry = {
                            #     'patient_id': parsed_data['patient_id'],
                            #     'scan_type': parsed_data['scan_type'],
                            #     'datetime': parsed_data['datetime'],
                            #     'filename': parsed_data['filename'],
                            #     'file_path': file_path,
                            #     'slice_number': None
                            # }
                            # all_data.append(entry)

    # Convert the data into a DataFrame
    df = pd.DataFrame(all_data)

    # Save the data into a CSV file
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")
    else:
        print("No valid DICOM files found. No metadata saved.")

    return df

# ############################### Main Execution ##################################

if __name__ == "__main__":
    # Process the directories and get the DataFrame
    df = process_folders(base_dir)

    # Display the first few rows
    print(df.head())
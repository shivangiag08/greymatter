import os
import re
from datetime import datetime
import pandas as pd

base_dir = "/Users/Agaaz/Downloads/ADNI_nimrat"  # Adjust this to your actual directory
output_path = "/Users/Agaaz/Downloads/zip10_metadata_from_dcm1.csv"

def parse_dicom_filename(file_name):
    # Regex patterns
    patient_id_pattern = r"(\d{3}_S_\d{4})"  # Matches patient ID format like 003_S_6644
    datetime_pattern = r"(\d{8})(\d{6})"  # Matches datetime in format YYYYMMDDHHMMSS

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
    # List to store parsed data
    all_data = []

    # Traverse directories with os.walk()
    for root, dirs, files in os.walk(base_dir):
        if 'I' in os.path.basename(root):  # Only look at folders containing DICOM files
            # Loop through files and try parsing the DICOM filenames
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    
                    # Parse the filename and append to the data list
                    parsed_data = parse_dicom_filename(file_name)
                    if parsed_data and parsed_data['patient_id']:  # Ensure patient_id exists
                        parsed_data['file_path'] = file_path  # Add full file path to the data
                        all_data.append(parsed_data)

    # Convert the data into a DataFrame
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file for further analysis
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")
    else:
        print("No valid DICOM files found. No metadata saved.")

    return df

# Process the directories and get the DataFrame
df = process_folders(base_dir)

# Display the first few rows
print(df.head())

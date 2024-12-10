import os
import re
from datetime import datetime
import pandas as pd


base_dir = "/Users/nimratkk/Documents/Projects/XAIforAD/ADNI"  # Adjust this to your actual directory
output_path = r"/Users/nimratkk/Documents/Projects/XAIforAD/zip10_metadata_from_dcm.csv"


def parse_dicom_filename(file_name):
    # Regex to extract datetime in format YYYYMMDDHHMMSS
    datetime_pattern = r"(\d{8})(\d{6})"  # Matches YYYYMMDDHHMMSS

    # Split the filename by underscores ('_')
    parts = file_name.split('_')

    # Extract patient ID
    patient_id = parts[1] + "_" + parts[2]  # e.g., 003_S_6644

    # Extract scan type
    scan_type = parts[3] if len(parts) >= 4 else None

    # Use regex to find the datetime in the filename
    match = re.search(datetime_pattern, file_name)
    if match:
        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMMSS
        datetime_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
        try:
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            datetime_obj = None
    else:
        datetime_obj = None

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
            # print(f'    Processing directory: {root}')
            
            # Loop through files and try parsing the DICOM filenames
            for file_name in files:
                if file_name.endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    
                    # Parse the filename and append to the data list
                    parsed_data = parse_dicom_filename(file_name)
                    if parsed_data:
                        parsed_data['file_path'] = file_path  # Add full file path to the data
                        all_data.append(parsed_data)

    # Convert the data into a DataFrame
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file for further analysis
    df.to_csv(output_path, index=False)
    print("Metadata saved")

    return df

# Set the base directory

# Process the directories and get the DataFrame
df = process_folders(base_dir)

# Display the first few rows
print(df.head())
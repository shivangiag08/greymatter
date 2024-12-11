import pydicom
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import h5py
import logging

# ############################### Configuration ###################################
'''
###############################################################################
######################### Change these file paths #############################
###############################################################################
'''

# File paths
metadata_path = r"/Users/nimratkk/Documents/Projects/XAIforAD/30_patients_zip5.csv"          # Updated metadata path
hdf5_output_path = r"/Users/nimratkk/Documents/Projects/XAIforAD/processed_images_zip5.h5" # Using .h5 for HDF5 format
fail_path = r"//Users/nimratkk/Documents/Projects/XAIforAD/failed_files.csv"

# Batch size
BATCH_SIZE = 1000

# Target image size
TARGET_SIZE = (256, 256)

# ############################### Setup Logging ####################################
logging.basicConfig(
    filename='preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# ############################### Functions ########################################

def create_placeholder(target_size=TARGET_SIZE):
    """
    Creates a placeholder image (all zeros).

    Args:
        target_size (tuple): Desired image size.

    Returns:
        numpy.ndarray: Placeholder image.
    """
    return np.zeros(target_size, dtype=np.uint8)

def preprocess_single_frame(img, target_size=TARGET_SIZE):
    """
    Resizes and pads a single 2D image to the target size.

    Args:
        img (numpy.ndarray): Input image.
        target_size (tuple): Desired image size.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    h, w = img.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image to the new dimensions
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Add padding to make it the target size
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return padded_img

def load_and_preprocess_dicom(file_path, slice_number, target_size=TARGET_SIZE):
    """
    Loads a specific slice from a DICOM file and preprocesses it.

    Args:
        file_path (str): Path to the DICOM file.
        slice_number (int): The slice number to extract (1-based indexing).
        target_size (tuple): Desired image size.

    Returns:
        numpy.ndarray or None: Preprocessed image or None if processing fails.
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(file_path)
        
        # Convert DICOM pixel data to numpy array
        img = dicom_data.pixel_array

        # Handle multi-frame DICOMs
        if img.ndim == 3:
            frames = img.shape[0]
            if slice_number < 1 or slice_number > frames:
                raise ValueError(f"Slice number {slice_number} is out of range for file {file_path} with {frames} slices.")
            single_frame = img[slice_number - 1]  # 0-based indexing

            # If the frame has multiple channels, convert to grayscale
            if single_frame.ndim > 2:
                single_frame = np.mean(single_frame, axis=-1).astype(single_frame.dtype)

            # Preprocess the image
            processed_img = preprocess_single_frame(single_frame, target_size)
            return processed_img

        elif img.ndim == 2:
            if slice_number != 1:
                raise ValueError(f"Requested slice number {slice_number} for single-frame DICOM {file_path}.")
            processed_img = preprocess_single_frame(img, target_size)
            return processed_img

        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape} in file {file_path}")

    except Exception as e:
        # Log the exception with file path and slice number
        logging.error(f"Failed to process {file_path}, slice {slice_number}: {e}")
        return None

# ############################### Main Processing ##################################

def main():
    for n in range(6, 10):
        metadata_path = f"/Users/nimratkk/Documents/Projects/XAIforAD/30_patients_zip{n}.csv"          # Updated metadata path
        hdf5_output_path = f"/Users/nimratkk/Documents/Projects/XAIforAD/processed_images_zip{n}.h5" # Using .h5 for HDF5 format

        # Load the metadata
        metadata_df = pd.read_csv(metadata_path)
        total_metadata_entries = len(metadata_df)
        logging.info(f"Total metadata entries to process: {total_metadata_entries}")

        # Initialize HDF5 file
        with h5py.File(hdf5_output_path, 'w') as hdf5_file:
            # Create a dataset with unlimited rows and fixed image shape
            dset = hdf5_file.create_dataset(
                'images',
                shape=(0, TARGET_SIZE[0], TARGET_SIZE[1]),
                maxshape=(None, TARGET_SIZE[0], TARGET_SIZE[1]),
                dtype=np.uint8,
                chunks=True  # Enable chunking for better performance
            )

            # Initialize list for failed entries
            failed_entries = []

            # Initialize buffer for batch processing
            buffer = []

            # Initialize counters
            processed_entries = 0
            total_images_added = 0
            total_placeholders = 0

            # Process each metadata entry
            for idx, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Processing metadata entries"):
                file_path = row['file_path']
                slice_number = row['slice_number']

                # Check if the file exists
                if not os.path.exists(file_path):
                    logging.warning(f"File does not exist: {file_path}")
                    failed_entries.append({
                        'file_path': file_path,
                        'slice_number': slice_number,
                        'error': 'File not found'
                    })
                    # Add a placeholder
                    buffer.append(create_placeholder())
                    total_placeholders += 1
                    continue

                # Load and preprocess the specific slice
                processed_img = load_and_preprocess_dicom(file_path, slice_number, TARGET_SIZE)

                if processed_img is not None:
                    buffer.append(processed_img)
                    processed_entries += 1
                else:
                    # Processing failed, add a placeholder
                    buffer.append(create_placeholder())
                    failed_entries.append({
                        'file_path': file_path,
                        'slice_number': slice_number,
                        'error': 'Processing failed'
                    })
                    total_placeholders += 1

                # When buffer reaches BATCH_SIZE, write to HDF5 and clear buffer
                if len(buffer) >= BATCH_SIZE:
                    # Determine current size
                    current_size = dset.shape[0]
                    # Resize the dataset to accommodate new data
                    dset.resize(current_size + len(buffer), axis=0)
                    # Write the data
                    dset[current_size:current_size + len(buffer), :, :] = np.array(buffer, dtype=np.uint8)
                    # Log the write operation
                    logging.info(f"Wrote {len(buffer)} images to H5. Total images now: {current_size + len(buffer)}")
                    # Update the total images counter
                    total_images_added += len(buffer)
                    # Clear the buffer
                    buffer = []

            # Write any remaining data in buffer
            if buffer:
                current_size = dset.shape[0]
                dset.resize(current_size + len(buffer), axis=0)
                dset[current_size:current_size + len(buffer), :, :] = np.array(buffer, dtype=np.uint8)
                logging.info(f"Wrote remaining {len(buffer)} images to H5. Total images now: {current_size + len(buffer)}")
                total_images_added += len(buffer)
                buffer = []

        print(f"Preprocessing complete. Images saved as {hdf5_output_path}.")
        logging.info(f"Preprocessing complete. Processed entries: {processed_entries}, Images added: {total_images_added}, Placeholders added: {total_placeholders}")

        # Save the list of failed entries
        if failed_entries:
            failed_df = pd.DataFrame(failed_entries)
            failed_df.to_csv(fail_path, index=False)
            print(f"{len(failed_entries)} metadata entries failed to process. Details saved in '{fail_path}'.")
            logging.info(f"{len(failed_entries)} metadata entries failed to process. Details saved in '{fail_path}'.")
        else:
            print("All metadata entries processed successfully.")
            logging.info("All metadata entries processed successfully.")

# Execute the main function
if __name__ == "__main__":
    main()
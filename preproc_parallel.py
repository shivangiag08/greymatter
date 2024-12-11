import pydicom
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ############################### Configuration ###################################
'''
################################################################################################
################################################################################################
#########################  Change these file paths  ############################################
################################################################################################
################################################################################################
'''

# File paths
metadata_path = r'C:\Shivangi\college\Sem 5\Deep Learning\DL project\zip3_metadata_from_dcm.csv'
hdf5_output_path = r'D:/DL_DATASET/processed_images_zip3.h5'  # Using .h5 for HDF5 format
fail_path = r'C:\Shivangi\college\Sem 5\Deep Learning\DL project\failed_files_2.csv'

# Batch size
BATCH_SIZE = 100

# Target image size
TARGET_SIZE = (256, 256)

# Number of parallel workers (Adjust based on your system)
NUM_WORKERS = 4  # Start with 4 and adjust as needed

# ############################### Functions ########################################

def create_placeholder(target_size=TARGET_SIZE):
    """
    Creates a placeholder image (all zeros) with the target size.
    """
    return np.zeros(target_size, dtype=np.uint8)

def preprocess_single_frame(img, target_size=TARGET_SIZE):
    """
    Resizes and pads a single 2D image to the target size.
    """
    try:
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
    except Exception as e:
        raise RuntimeError(f"Error during preprocessing frame: {e}")

def load_and_preprocess_dicom(file_path, target_size=TARGET_SIZE):
    """
    Loads a DICOM file, handles multi-frame and multi-dimensional images,
    and preprocesses each frame.
    Returns a list of preprocessed images.
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(file_path)
        
        # Check if 'PixelData' exists
        if 'PixelData' not in dicom_data:
            raise ValueError("DICOM file does not contain PixelData.")
        
        # Convert DICOM pixel data to numpy array
        img = dicom_data.pixel_array
        
        # Handle multi-frame DICOMs
        if img.ndim == 3:
            # Assuming the first dimension is frames
            frames = img.shape[0]
            processed_frames = []
            for i in range(frames):
                single_frame = img[i]
                # If the frame has multiple channels, convert to grayscale
                if single_frame.ndim > 2:
                    single_frame = np.mean(single_frame, axis=-1).astype(single_frame.dtype)
                processed_frame = preprocess_single_frame(single_frame, target_size)
                processed_frames.append(processed_frame)
            return processed_frames  # Return list of processed frames
        elif img.ndim == 2:
            # Single-frame DICOM
            processed_img = preprocess_single_frame(img, target_size)
            return [processed_img]  # Return as a list for consistency
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")
    except Exception as e:
        # Propagate the exception to be handled in the main loop
        raise RuntimeError(f"Failed to process {file_path}: {e}")

def worker(file_path):
    """
    Worker function to process a single DICOM file.
    Returns a tuple (processed_images, error_message).
    If processing is successful, error_message is None.
    If processing fails, processed_images is None.
    """
    try:
        processed_imgs = load_and_preprocess_dicom(file_path)
        return (processed_imgs, None)
    except Exception as e:
        return (None, str(e))

# ############################### Main Processing ##################################

def main():
    # Load the metadata
    metadata_df = pd.read_csv(metadata_path)
    
    # Initialize HDF5 file
    with h5py.File(hdf5_output_path, 'w') as hdf5_file:
        # Create a dataset with unlimited rows and fixed image shape
        # Adjust dtype as needed (e.g., 'uint8' for images)
        dset = hdf5_file.create_dataset(
            'images',
            shape=(0, TARGET_SIZE[0], TARGET_SIZE[1]),
            maxshape=(None, TARGET_SIZE[0], TARGET_SIZE[1]),
            dtype=np.uint8,
            chunks=True  # Enable chunking for better performance
        )
        
        # Initialize list for failed files
        failed_files = []
        
        # Initialize buffer for batch processing
        buffer = []
        
        # Initialize the ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all tasks to the executor
            futures = {executor.submit(worker, row['file_path']): row['file_path'] for index, row in metadata_df.iterrows()}
            
            # Iterate over the completed futures as they finish
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DICOM files"):
                dicom_file = futures[future]
                try:
                    processed_imgs, error = future.result()
                    if error:
                        print(f"Error processing {dicom_file}: {error}")
                        failed_files.append((dicom_file, error))
                        # Add a single placeholder
                        buffer.append(create_placeholder())
                    else:
                        buffer.extend(processed_imgs)  # Add all frames to the buffer
                except Exception as e:
                    print(f"Unhandled exception for {dicom_file}: {e}")
                    failed_files.append((dicom_file, str(e)))
                    # Add a single placeholder
                    buffer.append(create_placeholder())
                
                # When buffer reaches BATCH_SIZE, write to HDF5 and clear buffer
                if len(buffer) >= BATCH_SIZE:
                    # Determine current size
                    current_size = dset.shape[0]
                    # Resize the dataset to accommodate new data
                    dset.resize(current_size + len(buffer), axis=0)
                    # Write the data
                    dset[current_size:current_size + len(buffer), :, :] = np.array(buffer, dtype=np.uint8)
                    # Clear the buffer
                    buffer = []
        
        # After all futures are processed, write any remaining data in buffer
        if buffer:
            with h5py.File(hdf5_output_path, 'a') as hdf5_file:  # Open in append mode
                dset = hdf5_file['images']
                current_size = dset.shape[0]
                dset.resize(current_size + len(buffer), axis=0)
                dset[current_size:current_size + len(buffer), :, :] = np.array(buffer, dtype=np.uint8)
        
    print(f"Preprocessing complete. Images saved as {hdf5_output_path}.")
    
    # Save the list of failed files
    if failed_files:
        failed_df = pd.DataFrame(failed_files, columns=['file_path', 'error'])
        failed_df.to_csv(fail_path, index=False)
        print(f"{len(failed_files)} files failed to process. Details saved in '{fail_path}'.")
    else:
        print("All files processed successfully.")

if __name__ == "__main__":
    main()

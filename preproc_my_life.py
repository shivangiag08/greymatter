import pydicom
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import h5py

# ############################### Configuration ###################################
'''
################################################################################################
################################################################################################
#########################  Change these file paths  ############################################
################################################################################################
################################################################################################
'''

# File paths
metadata_path = r"/Users/nimratkk/Documents/Projects/XAIforAD/zip10_metadata_from_dcm.csv"
hdf5_output_path = r"/Users/nimratkk/Documents/Projects/XAIforAD/processed_images_zip10.h5"  # Using .h5 for HDF5 format
fail_path = r"/Users/nimratkk/Documents/Projects/XAIforAD/failed_files.csv"

# Batch size
BATCH_SIZE = 1000

# Target image size
TARGET_SIZE = (256, 256)

# Placeholder image (e.g., all zeros)
def create_placeholder(target_size=TARGET_SIZE):
    return np.zeros(target_size, dtype=np.uint8)

# ############################### Functions ########################################

def preprocess_single_frame(img, target_size=TARGET_SIZE):
    """
    Resizes and pads a single 2D image to the target size.
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

def load_and_preprocess_dicom(file_path, target_size=TARGET_SIZE):
    """
    Loads a DICOM file, handles multi-frame and multi-dimensional images,
    and preprocesses each frame.
    Returns a list of preprocessed images.
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(file_path)
        
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
        
        # Process files with tqdm progress bar
        for idx, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Processing DICOM files"):
            dicom_file = row['file_path']
            
            # Check if the file exists
            if not os.path.exists(dicom_file):
                print(f"File does not exist: {dicom_file}")
                failed_files.append((dicom_file, "File not found"))
                # Optionally, add a placeholder
                buffer.append(create_placeholder())
                continue
            
            try:
                processed_imgs = load_and_preprocess_dicom(dicom_file)
                buffer.extend(processed_imgs)  # Add all frames to the buffer
            except Exception as e:
                print(e)
                failed_files.append((dicom_file, str(e)))
                # Optionally, add a placeholder
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
        
        # Write any remaining data in buffer
        if buffer:
            current_size = dset.shape[0]
            dset.resize(current_size + len(buffer), axis=0)
            dset[current_size:current_size + len(buffer), :, :] = np.array(buffer, dtype=np.uint8)
            buffer = []
    
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
import h5py
import pandas as pd

for n in range(6, 10):
    # Paths
    metadata_csv_path = f"/Users/nimratkk/Documents/Projects/XAIforAD/30_patients_zip{n}.csv"
    hdf5_file_path = f"/Users/nimratkk/Documents/Projects/XAIforAD/processed_images_zip{n}.h5"

    # Load metadata
    metadata_df = pd.read_csv(metadata_csv_path)
    total_metadata_entries = len(metadata_df)

    # Open H5 file and count images
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        total_h5_images = hdf5_file['images'].shape[0]

    print(f"Total metadata entries: {total_metadata_entries}")
    print(f"Total images in H5 file: {total_h5_images}")

    if total_metadata_entries == total_h5_images:
        print("SUCCESS: The H5 file and metadata are perfectly aligned.")
    else:
        print("ERROR: Mismatch detected between H5 file and metadata.")
        print(f"Difference: {abs(total_h5_images - total_metadata_entries)} images")

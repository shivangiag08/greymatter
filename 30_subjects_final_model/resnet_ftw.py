# command to run: python3 resnet_ftw.py --h5_file /path/to/your_file.h5 --csv_file /path/to/your_file.csv

import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torchvision import models, transforms
from PIL import Image
import argparse

# ############################### Configuration ###################################
'''
###############################################################################
######################### Change these file paths #############################
###############################################################################
'''

# List of specific patient IDs
selected_patient_ids = [
    "098_S_4018", "098_S_4017", "116_S_1271", "031_S_0294", "031_S_4021", 
    "023_S_4020", "031_S_4024", "099_S_4022", "116_S_4010", "037_S_4028",
    "024_S_4084", "067_S_4782", "011_S_4827", "014_S_2185", "014_S_4401", 
    "022_S_6069", "041_S_4060", "041_S_4138", "041_S_4143", "041_S_4874",
    "011_S_0002", "011_S_0003", "011_S_0005", "011_S_0008", "022_S_0007", 
    "100_S_0015", "023_S_0030", "023_S_0031", "011_S_0016", "073_S_4393",
    '941_S_6499' '016_S_6931' '018_S_2155' '082_S_1119' '027_S_0835','116_S_1243'
]

# Define training and testing patient IDs
train_patient_ids = selected_patient_ids[:30]  # First 30 for training
test_patient_ids = selected_patient_ids[30:]   # Last 6 for testing

# Default output paths for feature maps
DEFAULT_TRAIN_FEATURES_PATH = "/Users/Agaaz/Downloads/train_features.npy"
DEFAULT_TEST_FEATURES_PATH = "/Users/Agaaz/Downloads/test_features.npy"

# ############################### Functions ########################################

def load_resnet50(device):
    """
    Loads the ResNet50 model pre-trained on ImageNet without the top layers.

    Args:
        device (torch.device): The device to load the model on.

    Returns:
        model (torch.nn.Module): ResNet50 model.
    """
    model = models.resnet50(pretrained=True)
    # Remove the final fully connected layer
    modules = list(model.children())[:-1]  # Remove the last fc layer
    model = torch.nn.Sequential(*modules)
    model.to(device)
    model.eval()
    return model

def preprocess_image(img_array):
    """
    Preprocesses the image for ResNet50:
    - Converts grayscale to RGB by duplicating channels.
    - Resizes to 224x224.
    - Applies ResNet50 preprocessing.

    Args:
        img_array (numpy.ndarray): Input image array of shape (256, 256).

    Returns:
        img_tensor (torch.Tensor): Preprocessed image tensor of shape (3, 224, 224).
    """
    # Convert to PIL Image
    if img_array.ndim == 2:
        # Grayscale to RGB by duplicating channels
        img_rgb = np.stack([img_array]*3, axis=-1)
    elif img_array.ndim == 3 and img_array.shape[2] == 1:
        img_rgb = np.concatenate([img_array]*3, axis=2)
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        img_rgb = img_array
    else:
        raise ValueError(f"Unsupported image shape: {img_array.shape}")
    
    img_pil = Image.fromarray(img_rgb.astype('uint8'), 'RGB')
    
    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ResNet50 expects these mean values
            std=[0.229, 0.224, 0.225]    # ResNet50 expects these std values
        )
    ])
    
    img_tensor = preprocess(img_pil)  # Shape: (3, 224, 224)
    
    return img_tensor

def extract_features(model, img_tensor, device):
    """
    Extracts feature vectors using the ResNet50 model.

    Args:
        model (torch.nn.Module): ResNet50 model.
        img_tensor (torch.Tensor): Preprocessed image tensor.
        device (torch.device): Device to perform computation on.

    Returns:
        features (numpy.ndarray): Extracted feature vector.
    """
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(device)  # Shape: (1, 3, 224, 224)
        features = model(img_tensor)  # Shape: (1, 2048, 1, 1)
        features = features.cpu().numpy().flatten()  # Shape: (2048,)
    return features

def load_existing_features(file_path):
    """
    Loads existing features from a .npy file if it exists.

    Args:
        file_path (str): Path to the .npy file.

    Returns:
        features (numpy.ndarray): Existing features or an empty array.
    """
    if os.path.exists(file_path):
        features = np.load(file_path)
        return features
    else:
        return np.empty((0, 2048), dtype=np.float32)  # Assuming ResNet50 features

def save_features(file_path, new_features):
    """
    Appends new features to an existing .npy file.

    Args:
        file_path (str): Path to the .npy file.
        new_features (numpy.ndarray): New features to append.
    """
    if os.path.exists(file_path):
        existing_features = np.load(file_path)
        combined_features = np.vstack((existing_features, new_features))
    else:
        combined_features = new_features
    np.save(file_path, combined_features)

# ############################### Main Processing ##################################

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description="Incrementally extract features from H5 and CSV files using ResNet50.")
    parser.add_argument('--h5_file', type=str, required=True, help='Exact path to the H5 file.')
    parser.add_argument('--csv_file', type=str, required=True, help='Exact path to the corresponding CSV file.')
    parser.add_argument('--train_features', type=str, default=DEFAULT_TRAIN_FEATURES_PATH, help='Path to train_features.npy.')
    parser.add_argument('--test_features', type=str, default=DEFAULT_TEST_FEATURES_PATH, help='Path to test_features.npy.')
    
    args = parser.parse_args()
    
    h5_path = args.h5_file
    csv_path = args.csv_file
    train_features_path = args.train_features
    test_features_path = args.test_features
    
    # Check if H5 and CSV files exist
    if not os.path.exists(h5_path):
        print(f"H5 file does not exist: {h5_path}")
        return
    if not os.path.exists(csv_path):
        print(f"CSV file does not exist: {csv_path}")
        return
    
    # Determine the device to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet50 model
    model = load_resnet50(device)
    print("ResNet50 model loaded.")
    
    # Load metadata
    metadata_df = pd.read_csv(csv_path)
    print(f"Loaded metadata from {csv_path}, total entries: {len(metadata_df)}")
    
    # Open H5 file
    with h5py.File(h5_path, 'r') as h5f:
        images = h5f['images'][:]  # Assuming dataset name is 'images'
        print(f"Loaded {images.shape[0]} images from {h5_path}")
    
    # Initialize lists to store new features
    new_train_features = []
    new_test_features = []
    
    # Iterate through each image and its metadata
    for idx in tqdm(range(len(metadata_df)), desc="Processing images"):
        meta = metadata_df.iloc[idx]
        patient_id = meta['patient_id']
        slice_number = meta['slice_number']
        
        img = images[idx]
        
        # Preprocess image
        try:
            img_tensor = preprocess_image(img)
        except Exception as e:
            print(f"Preprocessing failed for {h5_path}, slice {slice_number}: {e}")
            continue  # Skip this image
        
        # Extract features
        try:
            features = extract_features(model, img_tensor, device)
        except Exception as e:
            print(f"Feature extraction failed for {h5_path}, slice {slice_number}: {e}")
            continue  # Skip this image
        
        # Assign to train or test based on patient_id
        if patient_id in train_patient_ids:
            new_train_features.append(features)
        elif patient_id in test_patient_ids:
            new_test_features.append(features)
        else:
            print(f"Patient ID {patient_id} not in train or test lists. Skipping.")
            continue  # Skip if patient ID not recognized
    
    # Convert lists to numpy arrays
    if new_train_features:
        new_train_features = np.array(new_train_features, dtype=np.float32)
        print(f"Extracted {new_train_features.shape[0]} train features.")
    else:
        new_train_features = np.empty((0, 2048), dtype=np.float32)
        print("No new train features extracted.")
    
    if new_test_features:
        new_test_features = np.array(new_test_features, dtype=np.float32)
        print(f"Extracted {new_test_features.shape[0]} test features.")
    else:
        new_test_features = np.empty((0, 2048), dtype=np.float32)
        print("No new test features extracted.")
    
    # Append new features to existing .npy files
    if new_train_features.size > 0:
        save_features(train_features_path, new_train_features)
        print(f"Appended {new_train_features.shape[0]} features to {train_features_path}")
    else:
        print("No train features to append.")
    
    if new_test_features.size > 0:
        save_features(test_features_path, new_test_features)
        print(f"Appended {new_test_features.shape[0]} features to {test_features_path}")
    else:
        print("No test features to append.")
    
    print("Feature extraction and appending complete.")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from sklearn.preprocessing import LabelEncoder

# Set device (use Metal Performance Shaders if available, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Load metadata and diagnosis CSV files into Pandas DataFrames
metadata_df = pd.read_csv(r"/Users/Agaaz/Downloads/dicom_metadata.csv")
diagnosis_df = pd.read_csv("/Users/Agaaz/Downloads/DXSUM_04Dec2024.csv")

# 2. Load preprocessed images stored as a NumPy array
processed_images = np.load("/Users/Agaaz/Downloads/processed_images.npy")

# 3. Map diagnosis labels to patient IDs from the diagnosis DataFrame
metadata_df['DIAGNOSIS'] = metadata_df['patient_id'].apply(
    lambda x: diagnosis_df[diagnosis_df['PTID'] == x]['DIAGNOSIS'].values[0]
    if not diagnosis_df[diagnosis_df['PTID'] == x].empty else None
)

# Encode textual diagnosis labels into numeric labels
label_encoder = LabelEncoder()
metadata_df['label'] = label_encoder.fit_transform(metadata_df['DIAGNOSIS'])

# 4. Split the dataset into training (80%) and validation (20%) sets
train_df, val_df = train_test_split(metadata_df, test_size=0.2, stratify=metadata_df['label'], random_state=42)

# 5. Custom Dataset class to load MRI images and corresponding labels
class MRI_Dataset(Dataset):
    def __init__(self, metadata_df, images, transform=None):
        self.metadata_df = metadata_df  # DataFrame containing metadata
        self.images = images  # NumPy array of images
        self.transform = transform  # Transformations to apply on the images
    
    def __len__(self):
        return len(self.metadata_df)  # Total number of samples in the dataset
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)  # Access the image and convert to float32
        label = self.metadata_df.iloc[idx]['label']  # Retrieve the label for the sample
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, label  # Return the image and label pair

# 6. Define image transformations for preprocessing (e.g., resizing, normalization)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert the NumPy array to a PIL image
    transforms.Resize((224, 224)),  # Resize the image to 224x224 (ResNet input size)
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize for grayscale images
])

# 7. Create DataLoaders for training and validation datasets
train_dataset = MRI_Dataset(train_df, processed_images[train_df.index.values], transform=transform)
val_dataset = MRI_Dataset(val_df, processed_images[val_df.index.values], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 8. Initialize the ResNet-50 model with transfer learning
model = models.resnet50(weights='IMAGENET1K_V1')

# Modify the first convolutional layer to accept 1-channel (grayscale) input
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer for 3-class classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 3),  # Adjust the output layer for 3 classes
    nn.Softmax(dim=1)  # Apply Softmax activation for probabilities
)

# Move the model to the selected device (GPU or CPU)
model = model.to(device)

# 9. Define the loss function (CrossEntropyLoss) and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

# 10. Training loop
num_epochs = 2
best_val_acc = 0  # Variable to store the best validation accuracy

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Track the cumulative loss for the epoch
    correct_preds = 0  # Track the number of correct predictions
    total_preds = 0  # Track the total number of predictions
    
    start_epoch_time = time.time()  # Record the start time of the epoch

    # Training phase
    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        
        optimizer.zero_grad()  # Reset gradients
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy for the batch
        _, preds = torch.max(outputs, 1)  # Get predicted class indices
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)
        
        # Estimate remaining time for the epoch
        if batch_idx > 0:
            elapsed_time = time.time() - start_epoch_time
            batches_left = len(train_loader) - (batch_idx + 1)
            estimated_time_remaining = elapsed_time / (batch_idx + 1) * batches_left
            print(f"Estimated time remaining for epoch: {estimated_time_remaining:.2f} seconds", end="\r")

    train_loss = running_loss / len(train_loader)  # Average loss for the epoch
    train_acc = correct_preds / total_preds  # Training accuracy
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    correct_preds = 0
    total_preds = 0
    val_loss = 0  # Track cumulative validation loss
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
    
    val_loss = val_loss / len(val_loader)  # Average validation loss
    val_acc = correct_preds / total_preds  # Validation accuracy
    
    epoch_time = time.time() - start_epoch_time  # Calculate epoch duration
    print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds.")
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

print("Training complete.")  # Indicate the end of training

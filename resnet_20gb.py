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

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Load the merged metadata file
metadata_df = pd.read_csv(r"/Users/Agaaz/Downloads/dicom_metadata.csv")
diagnosis_df = pd.read_csv("/Users/Agaaz/Downloads/DXSUM_04Dec2024.csv")

# 2. Load preprocessed images (from .npy file)
processed_images = np.load("/Users/Agaaz/Downloads/processed_images.npy")

# 3. Extract diagnosis labels and map them to the subjects
metadata_df['DIAGNOSIS'] = metadata_df['patient_id'].apply(
    lambda x: diagnosis_df[diagnosis_df['PTID'] == x]['DIAGNOSIS'].values[0]
    if not diagnosis_df[diagnosis_df['PTID'] == x].empty else None
)

# Map diagnosis to numeric labels
label_encoder = LabelEncoder()
metadata_df['label'] = label_encoder.fit_transform(metadata_df['DIAGNOSIS'])

# 4. Split data into train and validation sets (80-20)
train_df, val_df = train_test_split(metadata_df, test_size=0.2, stratify=metadata_df['label'], random_state=42)

# 5. Create a custom dataset class
class MRI_Dataset(Dataset):
    def __init__(self, metadata_df, images, transform=None):
        self.metadata_df = metadata_df
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)  # Access and convert the image to float32
        label = self.metadata_df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 6. Define image transformations (for grayscale input)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 7. Create DataLoaders for train and validation sets
train_dataset = MRI_Dataset(train_df, processed_images[train_df.index.values], transform=transform)
val_dataset = MRI_Dataset(val_df, processed_images[val_df.index.values], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 8. Define the model (ResNet-50 with transfer learning)
model = models.resnet50(weights='IMAGENET1K_V1')

# Modify the first convolution layer to accept 1-channel input instead of 3
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer for 3-class classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 3),  # Output 3 classes
    nn.Softmax(dim=1)  # Softmax for multi-class classification
)

# Move the model to the GPU (if available)
model = model.to(device)

# 9. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

# 10. Training loop
num_epochs = 2
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    start_epoch_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels).item()
        total_preds += labels.size(0)
        
        # Estimate time remaining for the epoch
        if batch_idx > 0:
            elapsed_time = time.time() - start_epoch_time
            batches_left = len(train_loader) - (batch_idx + 1)
            estimated_time_remaining = elapsed_time / (batch_idx + 1) * batches_left
            print(f"Estimated time remaining for epoch: {estimated_time_remaining:.2f} seconds", end="\r")

    train_loss = running_loss / len(train_loader)
    train_acc = correct_preds / total_preds
    
    # Validation phase
    model.eval()
    correct_preds = 0
    total_preds = 0
    val_loss = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
    
    val_loss = val_loss / len(val_loader)
    val_acc = correct_preds / total_preds
    
    epoch_time = time.time() - start_epoch_time
    print(f"\nEpoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f} seconds.")
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

print("Training complete.")

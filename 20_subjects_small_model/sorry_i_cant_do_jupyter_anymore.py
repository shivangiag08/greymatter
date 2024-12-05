import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from sklearn.preprocessing import LabelEncoder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the merged metadata file
metadata_df = pd.read_csv(r"C:\Shivangi\college\Sem 5\Deep Learning\DL project\greymatter\20_subjects_small_model\test_merged_mri_metadata.csv")
diagnosis_df = pd.read_csv("D:/DL_DATASET/DXSUM_04Dec2024.csv")

# 2. Load preprocessed images (from .npy file)
processed_images = np.load("D:/DL_DATASET/processed_images.npy")

# 3. Extract diagnosis labels and map them to the subjects
# Mapping diagnosis to label for CN:1, MCI:2, AD:3
diagnosis_map = {
    1: 'CN',  # Cognitively Normal
    2: 'MCI',  # Mild Cognitive Impairment
    3: 'AD'  # Alzheimer's Disease / Dementia
}

# Merge the diagnosis information
metadata_df['DIAGNOSIS'] = metadata_df['patient_id'].apply(
    lambda x: diagnosis_df[diagnosis_df['PTID'] == x]['DIAGNOSIS'].values[0]
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
        # Get image and label (no need to access file_path anymore)
        image = self.images[idx]  # Accessing the image from the pre-loaded numpy array
        label = self.metadata_df.iloc[idx]['label']
        
        # Ensure the image is of type float32
        image = image.astype(np.float32)  # Convert the image to float32
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 6. Define image transformations (for grayscale input)
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL Image (required for torchvision transforms)
    transforms.Resize((224, 224)),  # Resize to 224x224 for VGG
    transforms.ToTensor(),  # Convert to tensor (will automatically convert to float32)
    transforms.Normalize(mean=[0.485], std=[0.229])  # Pretrained model normalization (adjusted for grayscale)
])

# 7. Create DataLoaders for train and validation sets
train_dataset = MRI_Dataset(train_df, processed_images[train_df.index.values], transform=transform)
val_dataset = MRI_Dataset(val_df, processed_images[val_df.index.values], transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# 8. Define the model (VGG with transfer learning)
model = models.vgg16(weights='IMAGENET1K_V1')

# Modify the first convolution layer to accept 1-channel input instead of 3
model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

# Freeze all layers except the last layer
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier for our 3-class problem
model.classifier[6] = nn.Sequential(
    nn.Linear(model.classifier[6].in_features, 3),  # Output 3 classes
    nn.Softmax(dim=1)  # Softmax for multi-class classification
)

# Move the model to the GPU (if available)
model = model.to(device)

# 9. Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

# 10. Training loop
num_epochs = 2  # Set number of epochs to 2
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    # Iterate over data
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
    
    # Calculate training loss and accuracy
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
    
    # Calculate validation loss and accuracy
    val_loss = val_loss / len(val_loader)
    val_acc = correct_preds / total_preds
    
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
    

print("Training complete.")

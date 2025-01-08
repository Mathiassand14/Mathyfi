# Det her er hvad chatten var promptet til at lave. 

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


class HasyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Path to the HASY folder containing subfolders for each symbol class.
        transform: torchvision transforms to apply to each image.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Collect all image files and their class names
        self.image_paths = []
        self.labels = []
        
        # List subfolders (each representing a symbol/class)
        class_folders = [d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))]
        
        # Sort class_folders to ensure consistent labeling
        class_folders.sort()
        
        # Map folder name -> class index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_folders)}
        
        for cls_name in class_folders:
            cls_path = os.path.join(root_dir, cls_name)
            # Collect image paths
            for img_file in glob.glob(os.path.join(cls_path, '*.png')):
                self.image_paths.append(img_file)
                self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open the image in grayscale mode
        image = Image.open(img_path).convert('L')  # 'L' means 1-channel grayscale
        
        if self.transform:
            image = self.transform(image)
        
        # Return (image_tensor, label)
        return image, label


# Paths
hasy_root = 'path/to/HASY/images'  # e.g., "HASY/images"

# Transforms: convert to tensor, optionally normalize
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),       # or 64x64, depending on your preference
    transforms.ToTensor(),
    # Optional: transforms.Normalize((0.5,), (0.5,))  # If you want to normalize
])

# Instantiate the dataset (contains *all* samples for now)
full_dataset = HasyDataset(root_dir=hasy_root, transform=data_transform)

# Create train/val split
indices = np.arange(len(full_dataset))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=full_dataset.labels  # Keep class distribution
)

# Subset datasets
from torch.utils.data import Subset

train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)

# Number of classes (assuming we used class_folders in alphabetical order)
num_classes = len(full_dataset.class_to_idx)
print(f"Number of classes: {num_classes}")



# Her begynder det neurale netværk

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input is 1-channel grayscale
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # After two 2x2 poolings with a 32x32 input, feature maps are 64 x 8 x 8
        # (because 32->16->8 in each dimension)
        
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

model = SimpleCNN(num_classes=num_classes)
print(model)


# Trænings setup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# For tracking
num_epochs = 10



# Trænings loop

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    val_loss /= val_total
    val_acc = val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# Evaluering

# Final evaluation (re-run the validation step, or do a dedicated test set)
model.eval()
# ...

# Save model
torch.save(model.state_dict(), "hasy_cnn.pth")


# Eventuelle tweaks man kan lave

#transforms.RandomRotation(degrees=10),
#transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),

#transforms.Normalize((0.5,), (0.5,))


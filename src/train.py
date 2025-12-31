import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os

# 1. Configuration
DATA_DIR = './dataset'
MODEL_SAVE_PATH = 'robocon_model.pth'
NUM_CLASSES = 3  # Logo, Oracle, Random
BATCH_SIZE = 32
EPOCHS = 10

# 2. Data Preprocessing (Augmentation & Normalization)
data_transforms = {
    'train_red': transforms.Compose([
        # REMOVED: transforms.RandomInvert(p=1.0), 
        # REMOVED: transforms.Grayscale(num_output_channels=3),
        
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2), # Helpful for lighting changes
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ]),
    'val_red': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# 3. Load Data
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train_red', 'val_red']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
               for x in ['train_red', 'val_red']}

# 4. Define Model (MobileNetV2)
print("Downloading MobileNetV2...")
model = models.mobilenet_v2(weights='DEFAULT')

# Freeze early layers to keep previous knowledge (optional, speeds up training)
for param in model.features.parameters():
    param.requires_grad = False

# Modify Classifier for 3 classes
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Training Loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Starting training on {device}...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloaders['train_red']:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloaders['train_red']):.4f}")

# 6. Save PyTorch Model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
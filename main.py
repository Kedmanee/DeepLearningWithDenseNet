import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os

# Define the class labels
class_labels = ['Tiger', 'Lion', 'Cheetah', 'Leopard', 'Puma']

# Define the custom dataset class
class FelidaeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = self.get_file_list()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.get_label_from_filename(img_path)
        return image, label

    def get_file_list(self):
        file_list = []
        for class_label in class_labels:
            class_dir = os.path.join(self.root_dir, class_label)
            files = os.listdir(class_dir)
            files = [os.path.join(class_dir, file) for file in files if not file.startswith('.DS_Store')]
            file_list.extend(files)
        return file_list

    def get_label_from_filename(self, filename):
        for i, class_label in enumerate(class_labels):
            if class_label.lower() in filename.lower():
                return i
        return -1

# Define the transformations for preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = models.densenet121(pretrained=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, len(class_labels))
model = model.to(device)

# Load the training and validation datasets
train_dataset = FelidaeDataset(root_dir='./train_images', transform=preprocess)
valid_dataset = FelidaeDataset(root_dir='./valid_images', transform=preprocess)

# Define the dataloaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 1

for epoch in range(num_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    train_correct = 0
    valid_correct = 0

    # Training phase
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        train_correct += (predicted == labels).sum().item()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            valid_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            valid_correct += (predicted == labels).sum().item()
            print(f"Predict : {predicted}")
            print(f"Image : {labels}")

    # Calculate average losses and accuracies
    train_loss = train_loss / len(train_dataset)
    valid_loss = valid_loss / len(valid_dataset)
    train_accuracy = train_correct / len(train_dataset) * 100
    valid_accuracy = valid_correct / len(valid_dataset) * 100

    # Print epoch results
    print(f"Epoch: {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")
    print(f"Valid Loss: {valid_loss:.4f} | Valid Accuracy: {valid_accuracy:.2f}%")
    print("--------------------------------------------")

# After training, you can use the model for evaluation as shown in the previous code snippet.

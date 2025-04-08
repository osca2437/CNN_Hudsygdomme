import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
num_classes = 6
batch_size = 32
num_epochs = 15
learning_rate = 0.001
image_size = 128  # resizing images to 128x128

# Data transformations: resize, convert to tensor, and normalize
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the datasets from train and test directories
train_dataset = datasets.ImageFolder(root="dataset/train", transform=transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # assuming 3-channel images
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Compute the size after convolution and pooling
        self.fc_input_dim = 64 * (image_size // 8) * (image_size // 8)
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)  # outputs raw logits
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.fc_input_dim)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # binary cross entropy with logits
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Helper function: convert integer labels to one-hot vectors
def to_one_hot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes).float()

# Training and evaluation
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
        # Move data to device
        inputs = inputs.to(device)
        # Convert labels to one-hot vectors
        labels_onehot = to_one_hot(labels, num_classes).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels_onehot)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # For accuracy, choose the class with highest logit value
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels.to(device)).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # Evaluation on test data
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels_onehot = to_one_hot(labels, num_classes).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels_onehot)
            test_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            correct_test += (preds == labels.to(device)).sum().item()
            total_test += labels.size(0)
    
    epoch_test_loss = test_loss / total_test
    epoch_test_acc = correct_test / total_test
    test_losses.append(epoch_test_loss)
    test_accuracies.append(epoch_test_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs} -- "
        f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | "
        f"Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc:.4f}")


    # Plotting the training and testing loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plotting the training and testing accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Save the plots
plt.savefig("training_testing_plots.png")
print("Plots saved as training_testing_plots.png")



# Save the model
torch.save(model.state_dict(), "simple_cnn_model.pth")
print("Model saved as simple_cnn_model.pth")
# Load the model (for inference or further training)
# model.load_state_dict(torch.load("simple_cnn_model.pth"))
# model.eval()  # Set the model to evaluation mode if needed
# Example inference code (uncomment to use)
# with torch.no_grad():
#     sample_input = torch.randn(1, 3, image_size, image_size).to(device)  # Example input
#     output = model(sample_input)
#     predicted_class = torch.argmax(output, dim=1)
#     print(f"Predicted class: {predicted_class.item()}")

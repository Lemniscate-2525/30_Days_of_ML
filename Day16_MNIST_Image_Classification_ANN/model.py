import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
import time

import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix

from torch.utils.data import DataLoader, random_split

raw_train_data = torchvision.datasets.MNIST(root ='./data', train = True, download = True, transform = transforms.ToTensor())
#print(len(raw_train_data))

# EDA : 
#labels = [label for i, label in raw_train_data]
#plt.hist(labels, bins = 10)
#plt.title("Class Dist : ")
#plt.show()

#fig, axes = plt.subplots(2,5, figsize = (10,4))

#for i, ax in enumerate(axes.flat):
#   img, label = raw_train_data[i]
#   ax.imshow(img.squeeze(), cmap = 'gray')
#   ax.set_title(label)
#   ax.axis('off')

#plt.show()

# Data Preprocessing :
loader = DataLoader(raw_train_data, batch_size = 60000, shuffle = False)
data = next(iter(loader))[0]   # Shape: (60000, 1, 28, 28)

mean = data.mean()  # Mean and Std Dev of Data for Preprocessing later.
std = data.std()

print(mean, std)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean.item(),), (std.item(),))])

# Data Loading : 
train_dataset = torchvision.datasets.MNIST(root ='./data', train=True, transform = transform, download = True)
test_dataset = torchvision.datasets.MNIST(root ='./data', train = False, transform = transform)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 64)
test_loader = DataLoader(test_dataset, batch_size = 64)

# ANN Model : 
class ANN(nn.Module):   
  def __init__(self):
    super().__init__()

    self.fc1 = nn.Linear(784, 128)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)

    return x
    
# Training : 
model = ANN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Training Time : 
losses = []
train_accs = []
val_accs = []
val_losses = []

start_time = time.time()

epochs = 10
for epoch in range(epochs):
  epoch_loss = 0
  correct = 0
  total = 0

  model.train()

  for x,y in train_loader:
    outputs = model(x)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()

    preds = torch.argmax(outputs, dim = 1)
    correct += (preds == y).sum().item()
    total += y.size(0)

  train_acc = correct / total
  train_accs.append(train_acc)

  losses.append(epoch_loss)

  model.eval()
  val_loss = 0
  correct = 0
  total = 0

  with torch.no_grad():
    for x,y in val_loader:
      outputs = model(x)
      loss = criterion(outputs, y)

      val_loss += loss.item()

      preds = torch.argmax(outputs, dim = 1)
      correct += (preds == y).sum().item()
      total += y.size(0)

  val_acc = correct / total
  val_accs.append(val_acc)
  val_losses.append(val_loss)

  print(f"Epoch {epoch} | Loss : {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

train_time = time.time() - start_time

# Inference Latency and Metrics : 
model.eval()

all_preds = []
all_labels = []

inf_lat = time.time()

with torch.no_grad():
  for x,y in test_loader:
    outputs = model(x)
    preds = torch.argmax(outputs, dim = 1)

    all_preds.extend(preds.numpy())
    all_labels.extend(y.numpy())

inf_lat = time.time() - inf_lat
print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))

print("Training Time:", train_time)
print("Inference Time:", inf_lat)
print("Per Sample Latency:", inf_lat / len(test_dataset))

plt.plot(losses)
plt.title("Training Loss Curve : ")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Confusion Matrix Visualization : 
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# Loss Visualization : 
plt.figure(figsize = (12,5))

plt.subplot(1,2,1)
plt.plot(losses, label = "Train Loss")
plt.plot(val_losses, label = "Val Loss")
plt.legend()

plt.title("Loss : ")

# Accuracy Visualization : 
plt.subplot(1,2,2)
plt.plot(train_accs, label = "Train Acc")
plt.plot(val_accs, label = "Val Acc")
plt.legend()
plt.title("Accuracy : ")

plt.show()

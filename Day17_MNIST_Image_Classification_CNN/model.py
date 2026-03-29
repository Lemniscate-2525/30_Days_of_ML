# Imports : 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import time

from torch.utils.data import DataLoader, random_split

raw_train_data = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())

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
data = next(iter(loader))[0]

mean = data.mean()   # Mean and std dev for Scaling.
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

# CNN Model : 
class CNN(nn.Module):
  def __init__ (self):
    super().__init__()

    self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, padding = 1)
    self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)

    self.relu = nn.ReLU()
    self.pool = nn.MaxPool2d(2, 2)

    self.fc1 = nn.Linear(32*7*7, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))

    x = x.view(x.size(0), -1)

    x = self.relu(self.fc1(x))
    x = self.fc2(x)

    return x

# Model, Training Time, Training & Validation Accuracy : 
model = CNN()

crit = nn.CrossEntropyLoss()  # Loss fn.
opt = optim.Adam(model.parameters(), lr = 0.001)  # Adam Goat.

losses = []
train_accs = []
val_accs = []
val_losses = []

epochs = 10

start_time = time.time()

for epoch in range(epochs):
  epoch_loss = 0
  corr = 0
  tot = 0

  model.train()

  for x,y in train_loader:
    outputs = model(x)
    loss = crit(outputs, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    epoch_loss += loss.item()

    preds = torch.argmax(outputs, dim = 1)
    corr += (preds == y).sum().item()
    tot += y.size(0)

  train_acc = corr/tot
  train_accs.append(train_acc)
  losses.append(epoch_loss)

  model.eval()
  val_loss = 0
  corr = 0
  tot = 0

  with torch.no_grad():
    for x,y in val_loader:
      outputs = model(x)
      loss = crit(outputs, y)

      val_loss += loss.item()

      preds = torch.argmax(outputs, dim = 1)
      corr += (preds == y).sum().item()
      tot += y.size(0)

  val_acc = corr/tot
  val_accs.append(val_acc)
  val_losses.append(val_loss)

  print(f"Epoch {epoch} | Loss : {epoch_loss :.4f} | Train Accuracy : {train_acc:.4f} | Validation Accuracy : {val_acc:.4f}")

train_time = time.time() - start_time

# Model Evaluation, Metrics, Inference Latency : 
model.eval()

all_preds = []
all_labels = []

start_inf = time.time()

with torch.no_grad():
    for x, y in test_loader:
        outputs = model(x)
        preds = torch.argmax(outputs, dim = 1)

        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

inf_time = time.time() - start_inf

print("\nClassification Report :")
print(classification_report(all_labels, all_preds))

print("\nConfusion Matrix :")
print(confusion_matrix(all_labels, all_preds))

print("\nTraining Time :", train_time)
print("Inference Time :", inf_time)
print("Per Sample Latency :", inf_time / len(test_dataset))

# Loss & Accuracy Visualization : 
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)

plt.plot(losses, label = "Train Loss")
plt.plot(val_losses, label = "Val Loss")

plt.legend()
plt.title("Loss : ")

plt.subplot(1,2,2)

plt.plot(train_accs, label = "Train Acc")
plt.plot(val_accs, label = "Val Acc")

plt.legend()
plt.title("Accuracy : ")

plt.show()

# Confusion Matrix Visualization : 
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize = (8,6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')

plt.title("Confusion Matrix : ")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

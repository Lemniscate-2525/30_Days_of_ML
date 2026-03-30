import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
import seaborn as sns

import time
import copy

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

# Hardware : 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")

# Data Preprocessing : 
data_transforms = {'train' :  transforms.Compose([transforms.Resize((224, 224)), 
transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),   
                  
'val' : transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}

# EDA : 
data_dir = 'chest_xray'
image_datasets = {x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 32, shuffle = True, num_workers = 2) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Classes :{class_names}")
print(f"Training Images : {dataset_sizes['train']} | Validation Images : {dataset_sizes['val']}")

# Model : 
model = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():  # Freezing Base Model.
  param.requires_grad = False

n_feat = model.fc.in_features
model.fc = nn.Linear(n_feat, 1)
model = model.to(device)

crit = nn.BCEWithLogitsLoss()

# Helper fn for Model Training : 
def train_model(model, crit, optimizer, num_epochs = 5):
  t = time.time()
  hist = {'train_loss' : [], 'val_loss' : [], 'train_acc' : [], 'val_acc' : []}
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0

  for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    for phase in['train', 'val']:
      if phase == 'train':
        model.train()
      else:
        model.eval()

      running_loss = 0.0
      running_corr = 0

      for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = model(inputs)
          loss = crit(outputs, labels)
          pred = (torch.sigmoid(outputs) > 0.5).float()

          if phase == 'train':
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corr += torch.sum(pred == labels.data)
      
      epoch_loss = running_loss/dataset_sizes[phase]
      epoch_acc = running_corr.double()/dataset_sizes[phase]

      hist[f'{phase}_loss'].append(epoch_loss)
      hist[f'{phase}_acc'].append(epoch_acc.item())

      print(f'{phase.capitalize()} Loss : {epoch_loss :.4f} Acc : {epoch_acc:.4f}')

      if phase == 'val' and epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    print()
  
  # Training Time : 
  time_elapsed = time.time() - t
  print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best Val Acc: {best_acc:4f}')

  model.load_state_dict(best_model_wts)
  return model, hist, time_elapsed

# Model Training Phase 1 : 
print("Feature Extraction : ")
optim_p1 = optim.Adam(model.fc.parameters(), lr = 1e-3)
model, hist1, t1 = train_model(model, crit, optim_p1, num_epochs = 5)

# Model Training Phase 2 : 
print("Fine Tuning : ")
for param in model.parameters():
  param.requires_grad = True

optim_p2 = optim.Adam([{'params' : model.conv1.parameters(), 'lr' : 1e-5},
{'params': model.layer1.parameters(), 'lr' : 1e-5},
{'params': model.layer2.parameters(), 'lr': 1e-5},
{'params': model.layer3.parameters(), 'lr': 1e-5},
{'params': model.layer4.parameters(), 'lr': 1e-5},
{'params': model.fc.parameters(), 'lr': 1e-4}
                       
])

model, hist2, t2 = train_model(model, crit, optim_p2, num_epochs = 5)

# Inference Latency : 
print("Calculating Inference Latency : ")
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).to(device)
start_inf = time.time()
with torch.no_grad():
    _ = model(dummy_input)
inf_latency = time.time() - start_inf

print(f"Per-sample Inference Latency : {inf_latency:.5f} seconds") 

# Loss & Accuracy Calculation : 
train_loss = hist1['train_loss'] + hist2['train_loss']
val_loss = hist1['val_loss'] + hist2['val_loss']
train_acc = hist1['train_acc'] + hist2['train_acc']
val_acc = hist1['val_acc'] + hist2['val_acc']
epochs = range(1, len(train_loss) + 1)

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 5))

# Loss Curve : 
ax1.plot(epochs, train_loss, label = 'Train Loss', color = 'blue')
ax1.plot(epochs, val_loss, label = 'Val Loss', color = 'red')
ax1.axvline(x = 5.5, color = 'gray', linestyle = '--', label = 'Thaw Base')
ax1.set_title('Training and Validation Loss : ')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

# Accuracy Curve : 
ax2.plot(epochs, train_acc, label = 'Train Acc', color = 'blue')
ax2.plot(epochs, val_acc, label = 'Val Acc', color = 'red')
ax2.axvline(x = 5.5, color = 'gray', linestyle = '--', label = 'Thaw Base')
ax2.set_title('Training and Validation Accuracy : ')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
plt.show()

# Confusion Matrix : 
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

print("\nClassification Report: ")
print(classification_report(y_true, y_pred, target_names=class_names))

# CM Visualization : 
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = class_names, yticklabels = class_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix on Validation Data')
plt.show()




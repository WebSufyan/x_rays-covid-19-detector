import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt

train_transformer = transforms.Compose([
    transforms.RandomRotation(50,expand=True),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

root = 'C:\\Users\\User\\OneDrive\\Bureau\\coding\\ARTIFICIAL INTELLIGENCE Specials\\x_rays covid-19 detector'

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transformer)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transformer)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True, num_workers=8)

# for i, (image, label) in enumerate(train_data):
#     break
# conv1 = nn.Conv2d(3, 32, 5)
# conv2 = nn.Conv2d(32, 16, 5)
# x = image.view(1, 3, 256, 256)
# conv1 = F.relu(conv1(x))
# pool1 = F.max_pool2d(conv1, 2, 2)
# conv2 = F.relu(conv2(pool1))
# pool2 = F.max_pool2d(conv2, 2, 2)




class covid19(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 16, 5, 1)
        self.fc1 = nn.Linear(61*61*16, 200)
        self.fc2 = nn.Linear(200, 60)
        self.fc3 = nn.Linear(60, 3)
        
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.drop(x)
        x = x.view(-1, 61*61*16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

gpu = torch.device('cuda:0')
torch.manual_seed(43)

#training phase
def train_model(train_loader, model):
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    train_corr = 0
    
    for batch, (x_train, y_train) in enumerate(train_loader):
        y_pred = model(x_train.to(gpu))
        cost = loss(y_pred, y_train.to(gpu))
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_predicted = (torch.argmax(y_pred, 1))
        t_corr = (train_predicted == y_train.to(gpu)).sum()
        train_corr += t_corr
        
    return y_pred, cost, train_corr.item()

#evaluating phase
def evaluate_model(test_loader, model):
    loss = nn.CrossEntropyLoss()
    test_corr = 0
    with torch.no_grad():
        for test_batch, (x_test, y_test) in enumerate(test_loader):
            y_val = model(x_test.to(gpu))
    
            test_predicted = torch.argmax(y_val, 1)
            b_corr = (test_predicted == y_test.to(gpu)).sum()
            test_corr += b_corr
        test_cost = loss(y_val, y_test.to(gpu))
        
        return y_val, test_cost, test_corr.item()
                

epochs = 20

train_losses = []
test_losses = []

train_correct = []
test_correct = []

if __name__ == '__main__':
    
    model = covid19().to(gpu)
    
    for epoch in tqdm(range(epochs)):
        y_pred, cost, train_corr = train_model(train_loader, model)
        y_val, test_cost, test_corr = evaluate_model(test_loader, model)
        
        if epoch % 1 == 0:
            print(f'training loss: {cost}, training accuracy: {(train_corr/len(train_data)):.2},\
                  test accuracy: {test_corr/(len(test_data)):.2}')
        
        train_losses.append(cost)
        train_correct.append(train_corr)
        test_losses.append(test_cost)
        test_correct.append(test_corr)
  
plt.plot(train_losses, label = 'train')
plt.plot(test_losses, label = 'validation')
plt.legend()

# testing model on new image

# from PIL import Image

# image = Image.open('NORMAL(1267).jpg')
# image = test_transformer(image)

# class_name = train_data.classes

# predict = model(image.view(-1, 3, 256, 256).to(gpu))
# predicted = torch.argmax(predict, 1)
# print(class_name[predicted])

# torch.save(model.state_dict(), 'covid-19_detector.pt')

#this dataset was imbalanced, 
#so i had to under-sample it to prevent overfitting for the major label

# model accuracy 95%
























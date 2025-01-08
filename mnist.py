
import torch
import torch.nn as nn
from torch.optim import SGD #source: https://gbhat.com/machine_learning/gradient_descent_learning_rates.html
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt

#this exercise
##https://github.com/lukepolson/youtube_channel/blob/main/Python%20Tutorial%20Series/pytorch4.ipynb

#data
#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

#import data
x, y  = torch.load('C:\\Users\\Usuario\\OneDrive\\models\\quarto_python\\data\\mnist\\training.pt')

#show info about data
x.shape  #input
y.shape  #target or label

#show image
plt.imshow(x[3].numpy())
plt.title(f'Number is {y[3].numpy()}')
plt.colorbar()

#pre processing - dummy variable or one hot encoding
y_trans = F.one_hot(y, num_classes = 10)
y_trans.shape

#turn image into a vector
x.view(-1,28**2).shape


#dataset object
class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x /255.
        self.y = F.one_hot(self.y, num_classes = 10).to(float)
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]

#get training and testing data
train_mnist = CTDataset('C:\\Users\\Usuario\\OneDrive\\models\\quarto_python\\data\\mnist\\training.pt')
test_mnist = CTDataset('C:\\Users\\Usuario\\OneDrive\\models\\quarto_python\\data\\mnist\\test.pt')

xs, ys = train_mnist[0:4]

ys.shape
xs.shape

#data loader - put dataset into a dataloader with small batch size 5 to 32
train_dl_mnist = DataLoader(train_mnist, batch_size = 5)

for x, y in train_dl_mnist:
    print(x.shape)
    print(y.shape)
    break

#batch size =5 60000/5= 12000
len(train_dl_mnist)

mdl_loss = nn.CrossEntropyLoss()

#build the network
class MnistNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 100)
        self.Matrix2 = nn.Linear(100, 50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()

f = MnistNeuralNet()

xs.shape

f(xs)

mdl_loss(f(xs), ys)


#training loop
def mdl_train(dl, f, n_epochs = 20):
    mdl_optimizer = SGD(f.parameters(), lr = 0.01)
    mdl_loss = nn.CrossEntropyLoss()
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dl)
        for i , (x, y) in enumerate(dl):
            mdl_optimizer.zero_grad()
            loss_value = mdl_loss(f(x), y)
            loss_value.backward()
            mdl_optimizer.step()
            #store training data
            epochs.append(epochs+i/N)
            losses.append(loss_value.item())
        return np.array(epochs), np.array(losses)

epoch_data, loss_data = mdl_train(train_dl_mnist, f)

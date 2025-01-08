#n <- 100
#x <- rnorm(n, mean = 0, sd = 1)
#y <- 3*x + rnorm(n, mean = 0, sd = 0.7)

#nn y is a response(target), x is the predictor (input feature)
#mdl spec - define a neural network model
#mdl loss and otpz - define a loss function and optimizer MSE
#mdl fit - using backpropag and optimization

#packages
import torch
import torch.nn as nn
import torch.optim as optim

#create data
n = 100
x = torch.normal(mean  = 0, std = 1, size = (n,)) #x with normal dist mean 0 and sd 1
y = 3*x + torch.normal(mean = 0, std = 0.7, size = (n,)) #generate y as 3*x + noise

#reshape x to be n,1 as we need it in a 2d tensor for nn
x = x.view(n,1)
y = y.view(n,1)


#mdl spec
class MyLinearModel(nn.Module):
    def __init__(self):
        super(MyLinearModel, self).__init__()
        self.linear = nn.Linear(1,1) # 1 input and 2 out

    def forward(self, x):
        return self.linear(x)

#mdl loss and optiz
model = MyLinearModel() 
criterion = nn.MSELoss()  #MSE loss
optimizer = optim.SGD(model.parameters(), lr = 0.01) #stochastic gradient descent

#mdl training    

#maxit
epochs = 1000 
for epoch in range(epochs):
    model.train()
#forwar pass compute predicted y by passing x to the model
    y_pred = model(x) 
#compute loss
    loss = criterion(y_pred, y)
#backward pass and optimization
    optimizer.zero_grad() #clear gradiend
    loss.backward() #compute grad backpropagation
    optimizer.step() #update weights

#print the loss every 100 epochs
    if(epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

#test the model after training

model.eval()
with torch.no_grad():
    y_test = model(x)
    print(f"Predicted: {y_test[:5]}\nActual: {y[:5]}")


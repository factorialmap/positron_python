import torch
import numpy as np
from matplotlib import pyplot as plt

#create vector x hours training, y skill level
x = np.array([1.,2.,3.,4.,5.,6.,7.,8.])
y = np.array([3.,3.,6.,7.,11.,11.,15.,15.])

#relationship between hours of study vs skill level
plt.scatter(x,y)
plt.xlabel("month")
plt.ylabel("skill")
plt.show()

#define weight
w = 0.

#define linear relationship
def Model(x):
    return w * x

#loss function
def MSE(y_pred, y):
    return((y_pred-y)**2).mean()

#define steps to reducing loss
n_epochs = 100
learning_rate = 0.001 #how quickly the line move around the points

w_tracker = []
loss_tracker = []


for epoch in range(n_epochs):
    y_pred = Model(x)
    loss = MSE(y_pred, y)
    loss_tracker.append(loss)
    
    grad = 1/len(x)*2*sum((w*x-y)*x)
    w -= learning_rate*grad
    w_tracker.append(w)

#check results

fig, (ax1, ax2) = plt.subplots(1,2, figsize =(12,5))

for i in range(len(w_tracker)):
    ax1.cla()
    ax2.cla()

    ax1.scatter(x,y, label = "true values")
    ax1.plot(x, w_tracker[i]*x, label = f"Prediction - Epoch No {i+1}")
    ax1.set_title(f"Weight = {w_tracker[i]:.2f}")
    ax1.grid()
    ax1.legend()
    ax2.plot(w_tracker, loss_tracker, label = "Loss Curve")
    ax2.scatter(w_tracker[i], loss_tracker[i], color = "navy", label = "Current point")
    ax2.set_title("Loss vs Weight")
    ax2.grid()
    ax2.legend()
    plt.pause(0.5)
    plt.draw()

plt.show()

Model(10.)

import torch
from matplotlib import pyplot as plt

x = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])
y = torch.tensor([3.,3.,6.,7.,11.,11.,15.,15.])

plt.scatter(x,y)
plt.xlabel("Hours of study")
plt.ylabel("Skill")

w = torch.tensor(0., requires_grad = True)

def Model(x):
    return w * x

def MSE(y_pred, y):
    return ((y_pred - y)**2).mean()

n_epochs = 100
learning_rate = 0.001

w_tracker = []
loss_tracker = []

for epoch in range(n_epochs):
    #forward pass
    y_pred = Model(x)
    loss = MSE(y_pred, y)
    loss_tracker.append(loss.item())
    
    #backward pass
    loss.backward()
    with torch.no_grad():
        w -= learning_rate*w.grad
    w_tracker.append(w.item())
    w.grad.zero_()

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,5))

for i in range(len(w_tracker)):
    ax1.cla()
    ax2.cla()

    ax1.scatter(x,y, label = "true values")
    ax1.plot(x, w_tracker[i]*x, label = f"Prediction  - Epoch No {i + 1}")
    ax1.set_title(f"Weight = {w_tracker[i]:2f}")
    ax1.grid()
    ax1.legend()

    ax2.plot(w_tracker, loss_tracker, label = "Loss Curve")
    ax2.scatter(w_tracker[i], loss_tracker[i], color = "red", label=f"w at Epoch No {i +1}")
    ax2.set_title("Loss vs Weight")
    ax2.grid()
    ax2.legend()





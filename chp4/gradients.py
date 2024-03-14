import torch
from matplotlib import pyplot as plt

def f(time, params):
    a,b,c = params
    return a * (time**2) + (b*time) + c

def mean_squared_error(preds, targets):
    return ((preds - targets) ** 2).mean().sqrt()

def show_preds(preds, ax=None):
    if ax is None: ax=plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, preds.cpu().detach().numpy(), color='red') 
    ax.set_ylim(-300, 100)

time = torch.arange(0, 20).float()
speed = torch.randn(20) * 3 + 0.75 * (time - 9.5) ** 2 + 1 

params = torch.randn(3).requires_grad_()
preds = f(time, params) 
show_preds(preds)

loss = mean_squared_error(preds, speed)
loss.backward()
print(params.grad * 1.5)

# plt.show()




# plt.scatter(time, speed)
# plt.show)


# def f(x):
#     return (x**2).sum()

# xt = torch.tensor([3., 5., 9.]).requires_grad_()
# yt = f(xt)
# yt.backward()
# print(xt)
# print(yt)
# print(xt.grad)

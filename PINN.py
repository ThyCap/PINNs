from FCN import FCN
from tools import *

import seaborn as sns

from PIL import Image

import torch
import torch.autograd as autograd         # computation graph

import time

# 'Convert to tensor and send to GPU'
# X_train_Nf = torch.from_numpy(X_train_Nf).float()
# X_train_Nu = torch.from_numpy(X_train_Nu).float()
# U_train_Nu = torch.from_numpy(T_train_Nu).float()
# X_test = torch.from_numpy(X_test).float()
# u = torch.from_numpy(u_true).float()
# f_hat = torch.zeros(X_train_Nf.shape[0],1)

# Define the partial differencial equation that drives the loss calculation
def partial_diff_equation(f, g):
    f_x_y = autograd.grad(f, g, torch.ones([g.shape[0], 1]), retain_graph=True, create_graph=True)[0]  # first derivative
    f_xx_yy = autograd.grad(f_x_y, g, torch.ones(g.shape), create_graph=True)[0]  # second derivative

    f_yy = f_xx_yy[:, [1]] # we select the 2nd element for y
    f_xx = f_xx_yy[:, [0]] # we select the 1st element for x
    
    u = f_xx + f_yy + internalHeatTensor # loss equation
    u = u.float()

    return u

# define domain
X, Y, T = generate_domain()

# define boundary conditions

PINN = FCN(layers, X_train_PDE, X_train_BC, T_train, partial_diff_equation)

'Neural Network Summary'
print(PINN)

params = list(PINN.parameters())

start_time = time.time()

optimizer = PINN.optimizer

optimizer.step(PINN.closure)

elapsed = time.time() - start_time
print('Training time: %.2f' % (elapsed))


''' Model Accuracy '''
error_vec, u_pred = PINN.test()

print('Test Error: %.5f' % (error_vec))

sns.heatmap(u_pred)

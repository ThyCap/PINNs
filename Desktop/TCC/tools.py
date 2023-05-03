import numpy as np
import torch
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

from pyDOE import lhs

#Set default dtype to float32
torch.set_default_dtype(torch.float64)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

## Define type of problem 
# 1. Simple diffusion in square 
# 2. Diffusion in square with internal heat
# 3. Diffusion in square with circular hole
# 4. Combination ?
hasInternalHeat = False
squareHasHole = False

# Setting Up Variables
T_low, T_mid, T_high = 0, 0.3, 1
R, x_circle, y_circle, N_circle = 0.5, 0.5, 0.5, 100

x_min, x_max, N_x = 0, 1, 100
y_min, y_max, N_y = 0, 1, 100

N_u = 100 # sampling size for BC 
N_f = 100_000 # sampling size for PDE domain

steps=10_000
lr=1e-1
layers = np.array([2,32,32,32,32,32,32,32,32,1]) #8 hidden layers

tolerance = 1e-6

#Domain bounds
lb = x_min # lower bound
ub = x_max # upper bound 

# Generate Data
def generate_domain():
    x = torch.linspace(x_min, x_max, N_x).view(-1,1)
    y = torch.linspace(y_min, y_max, N_y).view(-1,1)

    x = x.float()
    y = y.float()

    X, Y = torch.meshgrid(x.squeeze(1), y.squeeze(1))

    T = torch.rand((N_x, N_y),dtype = float)

    for i in range(N_x):
        for j in range(N_y):
            if x[i] == x_min:
                T[i][j] = T_low
            elif x[i] == x_max:
                T[i][j] = T_high

    T = T.float()

    return X, Y, T

X, Y, T = generate_domain()

# Define center circle
def generate_circle():
    # Perimeter points
    circle_x = np.array([x_circle + R*np.cos(theta) for theta in np.linspace(0, 2*np.pi, N_circle)])
    circle_y = np.array([y_circle + R*np.sin(theta) for theta in np.linspace(0, 2*np.pi, N_circle)])

    circle_X = torch.from_numpy(np.transpose(np.vstack((circle_x, circle_y))))

    circle_T = torch.ones((circle_X.shape[0], 1))*T_mid

    return circle_X, circle_T

circle_X, circle_T = generate_circle() 

def isInCircleTensor(X):
    boolList = []

    for elem in X:
        x, y = elem
        boolList.append((x - x_circle)**2 + (y - y_circle)**2 - R**2 < tolerance)

    return boolList

# Boundary Conditions 
def generate_BC(X, Y):
    # define boundary conditions zones:
    left_X = np.hstack((X[0,:][:,None], Y[0, :][:,None]))
    left_T = T[0,:][:,None]

    right_X = np.hstack((X[-1,:][:, None], Y[0,:][:, None]))
    right_T = T[-1,:][:,None]

    X_test = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    X_train = np.vstack([left_X, right_X])
    T_train = np.vstack([left_T, right_T])

    if squareHasHole:
        circle_X, circle_T = generate_circle()

        X_train = np.vstack([X_train, circle_X])
        T_train = np.vstack([T_train, circle_T])

    return X_train, T_train, X_test

X_train, T_train, X_test = generate_BC(X, Y)

# randomly choose N_u indices for training
idx = np.random.choice(X_train.shape[0], N_u, replace = False)

X_train_BC = X_train[idx, :]
T_train_BC = T_train[idx, :]

def generate_PDE():
    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    X_train_PDE = x_min + (x_max-x_min)*lhs(2,N_f) 

    X_train_PDE = X_train_PDE[not isInCircleTensor(X_train_PDE)]

    while X_train_PDE.shape[0] != N_f:
        currentN = X_train_PDE.shape[0]
        new_X_train_PDE = x_min + (x_max-x_min)*lhs(2,N_f - currentN)
        print(X_train_PDE)
        print(new_X_train_PDE.shape)
        X_train_PDE = np.vstack((X_train_PDE, new_X_train_PDE))

        X_train_PDE = X_train_PDE[not isInCircleTensor(X_train_PDE)]
    
    return X_train_PDE

print(generate_PDE())

# Internal Heat tensor
internalHeatTensor = torch.zeros(T.shape)







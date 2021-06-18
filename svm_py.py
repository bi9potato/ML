# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% 11
def Keras(xi,yi,sigma=0.1):
    diff = np.sum((xi-yi)*(xi-yi)/(2*sigma*sigma))
    return np.exp(-diff)
def svmPred(model,X):
    m1 = np.size(X,0)
    m2 = np.size(model["X"],0)
    p = np.zeros(m1,)
    pred = np.zeros(m1,)

    X1 = np.sum(X*X,1)
    X2 = np.sum(model["X"]*model["X"],1)
    X11 = np.repeat(X1,m2)
    X11 = np.reshape(X11,[m1,m2])
    X12 = np.repeat(X2,m1)
    X12 = np.reshape(X12,[m2,m1])

    K = X11+X12.T - 2*X@model["X"].T
    K = np.power(Keras(1,0,0.1),K)
    y = np.repeat(model["y"],m1)
    y = np.reshape(y,[m2,m1]).T
    K = y*K
    y = np.repeat(model["alphas"],m1)
    y = np.reshape(y,[m2,m1]).T
    K = y*K
    p = np.sum(K,1)

    pred[p>=0]=1
    pred[p<0] = 0
    return pred
#%%
# predict
def svmPredict(model, X):
    X1 = np.sum(X**2, 1)
    X2 = np.sum(model["X"]**2, 1)
    
    m1 = np.size(X1, 0)
    m2 = np.size(X2, 0)
    p = np.zeros(m1,)
    pred = np.zeros(m1,)
    
    X1 = np.repeat(X1,m2)                            
    X2 = np.repeat(X2,m1)
    X1 = np.reshape(X1,[m1,m2]) # bsx
    X2 = np.reshape(X2,[m2, m1])
    K = X1+X2.T - 2 * (X @ model['X'].T  ) # bsx
    diff = np.sum((1-0)*(1-0))/(2*0.1*0.1) # bsx
    K1 = sim = np.exp(-diff)
    K = np.power(K1,K)
    
    y = np.repeat(model['y'], m1).reshape(m2, m1)   
    K = y.T* K 
    alphas = np.repeat(model['alphas'], m1).reshape(m2, m1) 
    K = alphas.T * K            
    p = np.sum(K, 1)

    pred[p >= 0] =  1
    pred[p <  0] =  0
    
    return p
#%%
def myplot2(X, y, model):
    tx = np.linspace(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1) 
    ty = np.linspace(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1) 
    tx, ty = np.meshgrid(tx, ty) 
    shape = np.shape(tx) 
    tx = tx.reshape(-1)
    ty = ty.reshape(-1)
    txy = np.c_[tx, ty] 
    
    
    Z = svmPredict(model, txy) 
    
    
    X1 = X[y == 0]
    X2 = X[y == 1]
    plt.scatter(X1[:, 0], X1[:, 1], marker='*')
    plt.scatter(X2[:, 0], X2[:, 1], marker='o')
    tx = np.reshape(tx, shape) 
    ty = np.reshape(ty, shape) 
    Z =  np.reshape(Z,  shape) 
    plt.contourf(tx, ty, Z, [0.49, 0.51])
    return Z
#%%
from scipy.io import loadmat
import numpy as np

data = loadmat('ex6data2.mat')
X = data['X']
y = np.squeeze(data['y'])
y = np.int64(y)

C = 1
sigma = 0.1

(m ,n) = X.shape
y[y==0] = -1

alphas = np.zeros(m)
b=0
E = np.zeros(m)
passes=0
eta=0
L=0
H=0
tol = 1e-3

X2 = np.sum(X**2, axis=1)
X1 = np.repeat(X2,m);# 
X1 = np.reshape(X1,[m,m]) # bsx
K = X1+X1.T - 2 * (X @ X.T) # bsx
diff = np.sum((1-0)*(1-0))/(2*0.1*0.1) # bsx
K1 = sim = np.exp(-diff)# 
K = np.power(K1,K) #
max_passes = 5

# Train
print('\nTraining ...')
dots = 12

while passes < max_passes:
    num_changed_alphas = 0
    for i in range(m):
        E[i] = b + np.sum(alphas * y * K[:, i]) - y[i]
        
        if(y[i] * E[i] < -tol and alphas[i] < C) or (y[i] * E[i] > tol and alphas[i] > 0):
            j = np.ceil(m * np.random.rand())
            # j = 205
            while j == i:
                j = np.ceil(m * np.random.rand())
            j = int(j)-1 # 索引不能是float
            
            E[j] = b + np.sum(alphas * y * K[:, j]) - y[j]
            
            alpha_i_old = alphas[i];
            alpha_j_old = alphas[j];
            
            if (y[i] == y[j]):
                L = max(0, alphas[j] + alphas[i] - C)
                H = min(C, alphas[j] + alphas[i])
            else:
                L = max(0, alphas[j] - alphas[i])
                H = min(C, C + alphas[j] - alphas[i])

            if (L == H):
                continue
            
            eta = 2 * K[i,j] - K[i,i] - K[j,j]
            if (eta >= 0):
                continue
            
            alphas[j] = alphas[j] - (y[j] * (E[i] - E[j])) / eta
            
            alphas[j] = min (H, alphas[j]);
            alphas[j] = max (L, alphas[j]);
            
            if (abs(alphas[j] - alpha_j_old) < tol):
                alphas[j] = alpha_j_old
                continue
            
            alphas[i] = alphas[i] + y[i]*y[j]*(alpha_j_old - alphas[j]);
            
            b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) *  K[i,j]- y[j] * (alphas[j] - alpha_j_old) *  K[i,j]
            b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) *  K[i,j] - y[j] * (alphas[j] - alpha_j_old) *  K[j,j]
            
            if (0 < alphas[i] and alphas[i] < C):
                b = b1
            elif(0 < alphas[j] and alphas[j] < C):
                b = b2
            else:
                b = (b1+b2)/2
                
            num_changed_alphas = num_changed_alphas + 1
        
    if (num_changed_alphas == 0):
        passes = passes + 1
    else:
        passes = 0
            
    # matlab 171 行，少一些东西
    print('.', end='')
    dots = dots + 1
    if dots > 78:
        dots = 0
        print('\n')
    

print(' Done! \n\n')

idx = alphas>0

model = dict()
model["X"] = X[idx,:]
model["y"] = y[idx]
model["b"]  =b
model["alphas"] = alphas[idx]
model["w"] = (alphas*y)@X
#%%
# visualizeBoundary
import matplotlib.pyplot as plt 

pos = np.where(y == 1)
neg = np.where(y == -1)# 原为0
x1 = np.squeeze(X[pos, 0])
x2 = np.squeeze(X[pos, 1])
plt.scatter(x1, x2)
x11 = np.squeeze(X[neg, 0])
x22 = np.squeeze(X[neg, 1])
plt.scatter(x11, x22)

#%%
if (np.size(X, 1) == 1):
    X = X.T

m = np.size(X, 0)
p = np.zeros((m, 1))
pred = np.zeros((m, 1))
w = (alphas * y).T * X[:,1]

p2 = X[:,1] * w + b

pred[p >= 0] =  1
pred[p <  0] =  0

#%%
Z = myplot2(X, y, model)
#%%
p = svmPredict(model, X)





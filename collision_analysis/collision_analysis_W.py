import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from itertools import permutations, product

n = 10
p = 3

NN = 1000
alpha = 0.95

use_normal = True
F_grid_size = 5
H0 = False # Null hypothesis = the distribution is permutation invariant.

count = 0

def Num_Test_Points(n, p):
    return 2*(n*p)**p
    #return int(np.ceil(n**3))

def w_Fn(x, X, E):
    return np.sum((np.prod(X <= np.sort(x,axis=0), axis=1)\
        - np.prod(X <= x, axis=1))*E, axis=0)

perm_list = np.array(list(permutations(list(range(p)))))
delta_vec1 = np.array([1e-6, 0])
delta_vec2 = np.array([0, 1e-6])
delta_vec3 = np.array([1e-6, 1e-6])

for ii in range(NN):
    
    # F_prob = np.random.rand(*((F_grid_size,)*p))
    # if H0:
    #     F_prob_ = 0
    #     for pi in perm_list:
    #         F_prob_ += np.transpose(F_prob,axes=list(pi))
    #     F_prob = F_prob_
    # F_prob = F_prob/np.sum(F_prob)
    # Sampled_Sqs = np.random.choice(list(range(F_grid_size**2)), n, p=F_prob.reshape(F_grid_size**p))
    
    # X = (np.array([Sampled_Sqs%F_grid_size, Sampled_Sqs//F_grid_size]).T + np.random.rand(n, p))/F_grid_size
    
    X = np.random.rand(*(n,p))
    # if ((p==2) & (NN==1)):
    #     #plt.imshow(F_prob[::-1,:])
    #     #plt.figure()
    #     plt.scatter(X[:,0], X[:,1])
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    
    # X_ = np.vstack(X[:, perm_list])
    # X_ = np.maximum(np.expand_dims(X_,axis=1), np.expand_dims(X_,axis=0))
    # X_ = X_.reshape((X_.shape[0]*X_.shape[1],X_.shape[2]))
    # X_ = np.unique(X_, axis=0)
    
    X_ = np.array(list(product(X.reshape((X.size,)), repeat=p)))
    
    # X_ = X
    # Xsorted = X_[(X_ == np.sort(X_,axis=1))[:,0],:]
    # Xunsorted = X_[(X_ != np.sort(X_,axis=1))[:,0],:]
    # try:
    #     X_ = np.vstack((np.vstack(Xsorted[:,perm_list]), Xunsorted, np.sort(Xunsorted, axis=1)))
    # except:
    #     None
    # X_ = np.maximum(np.expand_dims(X_,axis=1), np.expand_dims(X_,axis=0))
    # X_ = X_.reshape((X_.shape[0]*X_.shape[1],X_.shape[2]))
    # X_ = np.unique(X_, axis=0)
    #X_ = np.vstack((X_,X_-delta_vec1,X_-delta_vec2,X_-delta_vec3))
    
    # X_ = np.vstack((X, np.sort(X, axis=1)))
    # X_ = np.maximum(np.expand_dims(X_,axis=1), np.expand_dims(X_,axis=0))
    # X_ = X_.reshape((X_.shape[0]*X_.shape[1],X_.shape[2]))
    # X_ = np.unique(X_, axis=0)
    # X_ = np.vstack((X_,X_-delta_vec1,X_-delta_vec2,X_-delta_vec3))
    
    E = np.random.normal(loc=0, scale=1, size=n)
    E_ = np.vstack([[E]]*X_.shape[0]).transpose((1,0))
    W = np.max(np.abs(np.sum((np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(X_,axis=1),axis=0), axis=2)\
        - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(X_,axis=0), axis=2))*E_, axis=0)),axis=0)#/np.sqrt(n)
    
    # XX = np.maximum(np.expand_dims(X,axis=1), np.expand_dims(X,axis=0))
    # XX = XX.reshape((XX.shape[0]*XX.shape[1],XX.shape[2]))
    # XX = np.unique(XX, axis=0)
    #     #X_ = X_[(X_ == np.sort(X_,axis=1))[:,0],:]
    # XX = np.vstack(XX[:, perm_list[1:,:]])
    # EE = np.vstack([[E]]*XX.shape[0]).transpose((1,0))
    # WW = np.max(np.sum((np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(XX,axis=1),axis=0), axis=2)\
    #     - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(XX,axis=0), axis=2))*EE, axis=0),axis=0)#/np.sqrt(n)

    NN_ = Num_Test_Points(n,p)
    # Sampled_Sqs = np.random.choice(list(range(F_grid_size**2)), NN_, p=F_prob.reshape(F_grid_size**p))
    # XX = (np.array([Sampled_Sqs%F_grid_size, Sampled_Sqs//F_grid_size]).T + np.random.rand(NN_, p))/F_grid_size
    XX = np.random.rand(*(NN_,p))
    
    # XX = np.vstack(X[:, perm_list])
    # XX = np.maximum(np.expand_dims(XX,axis=1), np.expand_dims(XX,axis=0))
    # XX = XX.reshape((XX.shape[0]*XX.shape[1],XX.shape[2]))
    # XX = np.unique(XX, axis=0)
    
    EE = np.vstack([[E]]*XX.shape[0]).transpose((1,0))
    WW = np.max(np.abs(np.sum((np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(XX,axis=1),axis=0), axis=2)\
        - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(XX,axis=0), axis=2))*EE, axis=0)),axis=0)#/np.sqrt(n)
    
    #count += (W-WW)/W
    #print((W-WW)/W)
    if WW <= W:
        count += 1
    else:
        plt.scatter(X[:,0], X[:,1])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        break
        
    print(f'Finished analyzing {ii+1} samples, no collision probability = {count/(ii+1)}.')
        
count = count/NN



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from itertools import permutations, product

n = 50
p = 3

NN = 1000
alpha = 0.95

use_normal = True
F_grid_size = 5
H0 = False # Null hypothesis = the distribution is permutation invariant.

count = 0

def Num_Test_Points(n, p):
    return int(np.ceil(n**3))

perm_list = np.array(list(permutations(list(range(p)))))

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
    #     plt.imshow(F_prob[::-1,:])
    #     plt.figure()
    #     plt.scatter(X[:,0], X[:,1])
    #     plt.xlim(0, 1)
    #     plt.ylim(0, 1)
    
    # #X_ = X[(X==np.sort(X,axis=1))[:,0],:]
    # #X_ = np.array(list(product(*X.T)))
    # X_ = np.maximum(*X[np.array(list(product(list(range(n)), repeat=p))).T,:])
    # #X_ = np.maximum(np.expand_dims(X,axis=1), np.expand_dims(X,axis=0))
    # #X_ = X_.reshape((X_.shape[0]*X_.shape[1],X_.shape[2]))
    # X_ = np.unique(X_, axis=0)
    # #X_ = X_[(X_ == np.sort(X_,axis=1))[:,0],:]
    # X_ = np.vstack(X_[:, perm_list])
    
    #X_ = np.array(list(product(X.reshape((X.size,)), repeat=p)))
    
    X_ = np.maximum(np.expand_dims(X,axis=1), np.expand_dims(X,axis=0))
    X_ = X_.reshape((X_.shape[0]*X_.shape[1],X_.shape[2]))
    X_ = np.unique(X_, axis=0)
    X_ = np.vstack(X_[:, perm_list[1:,:]])
    
    #X_ = np.vstack((X_,X_+np.array([1e-6,1e-6,1e-6])))
    T = np.max(np.abs(np.sum(np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(X_,axis=1),axis=0), axis=2)\
        - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(X_,axis=0), axis=2), axis=0)))/np.sqrt(n)
    
    NN_ = Num_Test_Points(n,p)
    # Sampled_Sqs = np.random.choice(list(range(F_grid_size**2)), NN_, p=F_prob.reshape(F_grid_size**p))
    # XX = (np.array([Sampled_Sqs%F_grid_size, Sampled_Sqs//F_grid_size]).T + np.random.rand(NN_, p))/F_grid_size
    XX = np.random.rand(*(NN_,p))
    
    TT = np.max(np.abs(np.sum(np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(XX,axis=1),axis=0), axis=2)\
        - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(XX,axis=0), axis=2), axis=0)))/np.sqrt(n)

    count += T/TT
    #if TT <= T:
    #    count += 1
        
    print(f'Finished analyzing {ii+1} samples, no collision probability = {count/(ii+1)}.')
        
count = count/NN

#list(product(list(range(3)),repeat=3))
# np.maximum(*X[np.array([[0,1,2],[0,1,1]]).T,:])

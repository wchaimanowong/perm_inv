import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import permutations, product
import json

# n_list = [25,50,100,200]
# p_list = [2,3]
n_list = [50,100,200,400,800,1600,3200]
p_list = [2,3]
q_list = [0.1,0.2,0.3,0.4]

NN = 1000 #1000
alpha = 0.95

eps = 0.001 # COBYLA
#eps = 0.05 # TNC

if len(q_list) > 1:
    result = np.zeros((len(p_list), len(n_list), len(q_list)))
else:
    result = np.zeros((len(p_list), len(n_list)))
error_report = []

for kk in range(len(n_list)):
    for jj in range(len(p_list)):
        
        n = n_list[kk]
        p = p_list[jj]
        
        nstarts = int(n/2)
        
        def visualize_T(X, eps, steps):
            x1 = np.linspace(0,1,steps)
            x2 = np.linspace(0,1,steps)
            graph = np.zeros((len(x1), len(x2)))
            for i in range(len(x1)):
                for j in range(len(x2)):
                    graph[i,j] = t_fn_smoothed(np.array([x1[i],x2[j]]), X, eps)
            plt.imshow(graph)
            
        def visualize_W(X, E, eps, steps):
            x1 = np.linspace(0,1,steps)
            x2 = np.linspace(0,1,steps)
            graph = np.zeros((len(x1), len(x2)))
            for i in range(len(x1)):
                for j in range(len(x2)):
                    graph[i,j] = w_fn_smoothed(np.array([x1[i],x2[j]]), X, E, eps)
            plt.imshow(graph)
            
        def cross_sect_W(X, E, steps, x_location=0.9):
            xi = np.linspace(0,1,steps)
            indices = np.array(range(steps))
            p = X.shape[1]
            mesh = np.array(list(product(xi, repeat=p)))
            mesh_indices = np.array(list(product(indices, repeat=p)))
            graph = np.zeros((len(xi),)*p)
            for i in range(mesh_indices.shape[0]):
                graph[tuple(mesh_indices[i])] = w_fn(mesh[i], X, E)
            plt.plot(xi,graph[(int(x_location*len(xi)),)*(p-1)])
            xlist = X.reshape((X.size,))
            for xc in xlist:
                plt.axvline(x=xc, color = 'red', linestyle = '--', alpha = 0.5)
        
        def step_smoothed(x, eps):
            return (np.tanh(x/eps)+1)/2
        
        def t_fn_smoothed(x, X, eps):
            return -np.abs(np.sum((np.prod(step_smoothed(np.sort(x,axis=0)-X, eps), axis=1)\
                                       - np.prod(step_smoothed(x-X, eps), axis=1)), axis=0))
        
        def w_fn_smoothed(x, X, E, eps):
            return -np.abs(np.sum((np.prod(step_smoothed(np.sort(x,axis=0)-X, eps), axis=1)\
                                       - np.prod(step_smoothed(x-X, eps), axis=1))*E, axis=0))
        
        def t_fn(x, X):
            return np.abs(np.sum((np.prod(X <= np.sort(x,axis=0), axis=1)\
                                       - np.prod(X <= x, axis=1)), axis=0))
        
        def w_fn(x, X, E):
            return np.abs(np.sum((np.prod(X <= np.sort(x,axis=0), axis=1)\
                                       - np.prod(X <= x, axis=1))*E, axis=0))
                
        perm_list = np.array(list(permutations(list(range(p)))))
        
        #construct the bounds in the form of constraints
        bnds = [[0,1],]*p
        cons = []
        for factor in range(len(bnds)):
            lower, upper = bnds[factor]
            l = {'type': 'ineq',
                 'fun': lambda x, lb=lower, i=factor: x[i] - lb}
            u = {'type': 'ineq',
                 'fun': lambda x, ub=upper, i=factor: ub - x[i]}
            cons.append(l)
            cons.append(u)
        
        count = np.zeros(len(q_list))
        for ii in range(NN):
            X = np.random.rand(*(n,p))
            
            starts = X.reshape((X.size,))[np.random.randint(0,n*p,(nstarts,p))]
            T = 0
            for start in starts:
                start = np.sort(start)[perm_list[np.random.randint(1,len(perm_list))]]
                res = minimize(t_fn_smoothed, x0 = start, args=(X, eps), constraints=cons, method='COBYLA', options={'maxiter':100})
                t_fn_val = t_fn(res.x, X)
                if t_fn_val > T:
                    T = t_fn_val
            T = T/np.sqrt(n)
                    
            XX = np.random.rand(*(int(0.5*((p**2)*n)**1.1),p))

            TT = np.max(np.abs(np.sum(np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(XX,axis=1),axis=0), axis=2)\
                - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(XX,axis=0), axis=2), axis=0)))
            TT = TT/np.sqrt(n)
                
            for ll in range(len(q_list)):
                if (T-TT > 0.5*(n_list[0]/n)**q_list[ll]):
                    count[ll] += 1
            print(f'Finished analyzing {ii+1} samples, P[T-T0 > 1/n^q] = {count/(ii+1)}% on average')
        result[jj][kk] = count/NN

        graph_coords = ""
        for jj_ in range(len(p_list)):
            for ll_ in range(len(q_list)):
                graph_coords += "q = " + str(q_list[ll_]) + ", p = " + str(p_list[jj_]) + ": \n"
                for kk_ in range(len(n_list)):
                    graph_coords += '(' + '{0:.3f}'.format(n_list[kk_]) + ',' + '{0:.3f}'.format(result[jj_][kk_][ll_]) + ') '
                graph_coords += "\n"
        with open('graph_result.txt', 'w') as f:
            f.write(graph_coords)
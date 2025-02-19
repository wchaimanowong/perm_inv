import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import permutations, product
from tqdm import tqdm

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

def main(n, p, alpha, H0, NN=1000, NN_=1000, eps=0.001, params={}):
    count = 0
    nstarts = int(n/2)
    
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
    
    for ii in range(NN):
        X = np.random.multivariate_normal(mean=params['mean'], cov=params['cov'], size=n)%1
        
        X_ = np.random.multivariate_normal(mean=params['mean'], cov=params['cov'], size=int(((p**2)*n)**1.1))
        T = np.max(np.abs(np.sum(np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(X_,axis=1),axis=0), axis=2)\
            - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(X_,axis=0), axis=2), axis=0)))/np.sqrt(n)
        
        E = np.random.normal(loc=0, scale=1, size=(n,NN_))
        E = np.vstack([[E]]*X_.shape[0]).transpose((1,0,2))
        W_list = np.max(np.abs(np.sum(np.expand_dims((np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(X_,axis=1),axis=0), axis=2)\
            - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(X_,axis=0), axis=2)), axis=2)*E, axis=0)),axis=0)/np.sqrt(n)
        W_list = np.sort(W_list)
        cW = W_list[int(alpha*NN_)]
        
        if H0:
            if T <= cW:
                count += 1
                
            print(f'Finished analyzing {ii+1} samples, coverage probability = {count/(ii+1)}.')
        else:
            if T > cW:
                count += 1
                
            print(f'Finished analyzing {ii+1} samples, power = {count/(ii+1)}.')
            
    return count/NN

if __name__ == '__main__':
    
    d_list = np.array([3,4,5])
    alpha_list = np.array([0.95, 0.99])
    #alpha_list = np.array([0.95,0.99])
    #shift_list = np.array([0.0,0.01,0.025,0.05])
    shift_list = np.array([0,0.1])
    cov_diag = 0.01
    
    result = np.zeros((len(shift_list), len(d_list), len(alpha_list)))
    
    
    for ii in range(len(shift_list)):
        for jj in range(len(d_list)):
            for kk in range(len(alpha_list)):
                shift = shift_list[ii]
                d = d_list[jj]
                alpha = alpha_list[kk]
                
                mean_vec = 0.5*np.ones(d)
                mean_vec[0] -= shift
    
                params={'mean':mean_vec, 'cov':np.eye(d)}
                if (np.abs(shift) == 0):
                    H0 = True
                else:
                    H0 = False
                result[ii][jj][kk] = main(n=100, p=d, alpha=alpha, H0=H0, params=params)
                
    # Make a LaTeX table
    latex = "\\begin{table}[ht!]\n\\centering\n\\begin{tabular}{@{}llllllllll@{}}\\toprule\n & Shift"
    for ii in range(len(shift_list)):
        latex += "& \\multicolumn{2}{c}{" + str(shift_list[ii]) + "}"
    latex += "\\\\ \\midrule\n $n$ & $\\alpha$"
    for ii in range(len(shift_list)):
        for kk in range(len(alpha_list)):
            latex += "& " + str(int(100*alpha_list[kk])) + "\%"
    latex += "\\\\ \\midrule\n"
    for jj in range(len(d_list)):
        latex += "\\multirow{2}{*}{" + str(d_list[jj]) + "} & Cov"
        for ii in range(len(shift_list)):
            for kk in range(len(alpha_list)):
                shift = shift_list[ii]
                if (np.abs(shift) == 0):
                    latex += " & $\\mathbf{" + '{0:.2f}'.format(result[ii][jj][kk]) + "}$"
                else:
                    latex += " & $" + '{0:.2f}'.format(1 - result[ii][jj][kk]) + "$"
        latex += "\\\\ \n"
        latex += "& Pow"
        for ii in range(len(shift_list)):
            for kk in range(len(alpha_list)):
                shift = shift_list[ii]
                if (np.abs(shift) == 0):
                    latex += " & $" + '{0:.2f}'.format(1 - result[ii][jj][kk]) + "$"
                else:
                    latex += " & $\\mathbf{" + '{0:.2f}'.format(result[ii][jj][kk]) + "}$"
        latex += "\\\\ \n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}"
    
    with open("perm_inv_test_dim_result.tex", "w") as f:
        f.write(latex)
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from itertools import permutations, product
from tqdm import tqdm

NN_=1000
alpha_list = [0.95, 0.99] # NOTE: Different convention to alpha in the paper (= 1-alpha_paper).
symmetrize_data = False
axis_wise_rescale = True

###############################################################################

# Load AA Data:

with open("ais.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=","))
data = np.array(data)
data = np.array(data[1:][:,:3], dtype=np.float32)

# Ionosphere

# with open("Ionosphere.csv", 'r') as f:
#     data = list(csv.reader(f, delimiter=","))
# data = np.array(data)
# data = np.array(data[1:][:,5:8], dtype=np.float32)

# Synthetic Circles

# with open("circles.csv", 'r') as f:
#     data = list(csv.reader(f, delimiter=","))
# data = np.array(data)
# data = np.array(data[1:][:,:2], dtype=np.float32)
# np.random.shuffle(data)
# data = (data[:200] + 10) % 20 # this data set is too big, do a random subsampling...

# Occupancy Estimation

#with open("occupancy.csv", 'r') as f:
#    data = list(csv.reader(f, delimiter=","))
#data = np.array(data)
#np.random.shuffle(data)
#data = np.array(data[1:501][:,11:14], dtype=np.float32)

with open("winequality-red.csv", 'r') as f:
    data = list(csv.reader(f, delimiter=";"))
data = np.array(data)
data = np.array(data[1:][:,:3], dtype=np.float32)
np.random.shuffle(data)
data = data[:202]

###############################################################################

def scatter_plot_data(data, idx1, idx2):
    X = data[:, idx1]
    Y = data[:, idx2]
    plt.scatter(X,Y)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

###############################################################################

p = data.shape[1]

# normalize data:
if not axis_wise_rescale:
    data = (data - np.min(data))/(np.max(data) - np.min(data))
else:
    for c in range(data.shape[1]):
        data[:,c] = (data[:,c] - np.min(data[:,c]))/(np.max(data[:,c]) - np.min(data[:,c]))

perm_list = np.array(list(permutations(list(range(p)))))

if symmetrize_data:
    data = data[:10]
    data = np.vstack(data[:, perm_list])
    data = data + np.random.normal(loc=0.0,scale=0.001,size=data.shape)
    data = data % 1.0

X = data
n = X.shape[0]

X_ = np.maximum(np.expand_dims(X,axis=1), np.expand_dims(X,axis=0))
X_ = X_.reshape((X_.shape[0]*X_.shape[1],X_.shape[2]))
X_ = np.unique(X_, axis=0)
X_ = np.vstack(X_[:, perm_list[1:,:]])

T = np.max(np.abs(np.sum(np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(X_,axis=1),axis=0), axis=2)\
                      - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(X_,axis=0), axis=2), axis=0)))/np.sqrt(n)
print(T)

X_ = np.vstack(X[:, perm_list])
X_ = np.maximum(np.expand_dims(X_,axis=1), np.expand_dims(X_,axis=0))
X_ = X_.reshape((X_.shape[0]*X_.shape[1],X_.shape[2]))
X_ = np.unique(X_, axis=0)

W_list = []
for ii in tqdm(range(NN_)):
    e = np.random.normal(loc=0,scale=1,size=n)
    e = np.vstack([[e]]*X_.shape[0]).transpose((1,0))
    W_list.append(np.max(np.abs(np.sum((np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(np.sort(X_,axis=1),axis=0), axis=2)\
                                               - np.prod(np.expand_dims(X,axis=1) <= np.expand_dims(X_,axis=0), axis=2))*e, axis=0)),axis=0)/np.sqrt(n))
W_list = np.array(W_list)
W_list = np.sort(W_list)

for alpha in alpha_list:
    cW = W_list[int(alpha*NN_)]

    print(" Done!")

    if T > cW:
        print(f"Reject the perm invariant hypothesis (H0) at {alpha} confidence level. T = {T}, cW = {cW}")
    else:
        print(f"Unable to reject the perm invariant hypothesis (H0) at {alpha} confidence level. T = {T}, cW = {cW}")
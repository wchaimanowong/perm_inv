Experimental code for the numerical studies in the AISTATS2025 paper: "Permutation Invariant Functions: Statistical Testing, Density Estimation, and Metric Entropy".

perm_inv_test.py and perm_inv_test_dim.py:

Demonstrate our multiplier bootstrap test of permutation invariance on a simulated data set.
The data set is generated from various multivariate normal distributions, we report the test results for various number of sample points n and dimensions d. 

perm_inv_test_data_analysis.py:

Demonstrate our multiplier bootstrap test of permutation invariance on an arbitrary dataset. Load your dataset into the 'data' variable and run the code. 
The real-world datasets in our numerical studies can be downloaded from UCI (see the reference in the paper).

perm_inv_pairwise.r:

A pairwise test of permutation invariance based on pHollBivSym, for comparison with our test in perm_inv_test_data_analysis.py.

perm_kernel_simulation.m:

The code for our permutation invariance density estimation.

collision_analysis:

Some experiments on several approaches for estimating the supremum in our test statistics.

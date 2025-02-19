Experimental code for the numerical experiments in the AISTATS2025 paper: "Permutation Invariant Multivariate Distribution Functions: Statistical Testing and Density Estimation".

perm_inv_test.py and perm_inv_test_dim.py:

Demonstrate our multiplier bootstrap test of permutation invariance on a simulated data set.
The data set is generated from various multivariate normal distributions, we report the test results for various number of sample points n and dimensions d. 

perm_inv_test_data_analysis.py:

Demonstrate our multiplier bootstrap test of permutation invariance on real-world data sets which can be downloaded from UCI.

perm_inv_pairwise.r:

A pairwise test of permutation invariance based on pHollBivSym, for comparison with our test in perm_inv_test_data_analysis.py.

perm_kernel_simulation.m:

The code for our permutation invariance density estimation.

collision_analysis:

Some experiments on several approaches for estimating the supremum in our test statistics.

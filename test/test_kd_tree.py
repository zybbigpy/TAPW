import numpy as np
import time 

from sklearn.neighbors import KDTree

print("="*100)
rng = np.random.RandomState(0)
X = rng.random_sample((10000, 2))  
Y = rng.random_sample((1000, 2))         

t1 = time.time()
tree = KDTree(X) 
ind = tree.query_radius(Y, r=0.4)               
t2 = time.time()

print("t2 - t1 =", t2 -t1)
print("====== finish KDTree test ======")
print("="*100)
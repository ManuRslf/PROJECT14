from scipy.stats import qmc
import numpy as np
import math as m
#lower bounds
lb = np.array([65, 0, 0, 0.03, 0.94, m.exp(3.0)*10**-10, 0.020, 0.05])

#upper bound
ub = np.array([75, 1, 0.5, 0.09, 0.99,  m.exp(3.5)*10**-10, 0.025, 0.30])

#initialise the dimension of points
sample = qmc.LatinHypercube(d = 8)

#nb of points
sampler = sample.random(n = 100000, workers=-1)

#gen points
np.save("Data/D_Params.npy", qmc.scale(sampler, lb, ub))

print("Params generated and saved")


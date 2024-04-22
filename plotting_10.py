import matplotlib.pyplot as plt
import numpy as np


raw = np.load("Data/TData_BB.npy")
x = [j for j in range(len(raw[1]))]

plt.figure(figsize=(10,10))

for i in range(10):
  plt.plot(x, raw[i])
  
plt.legend([f"$r = {raw[i][2]}" for i in range(10)], fontsize="9",loc='upper right')
plt.xlabel('$l$')
plt.ylabel('BB')

plt.grid()
plt.show()


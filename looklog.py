import numpy as np
import matplotlib.pyplot as plt
with open('log.txt') as f:
    data = np.loadtxt(f,delimiter = ",")
data[:,2][data[:,2] > 1] = 1
plt.figure(figsize=(20,20))
plt.grid()
plt.plot(data[:,0], data[:,2])
plt.plot(data[:,0], data[:,3])
plt.show()
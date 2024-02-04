from pic2func import function_from_picture

data = function_from_picture("test.png")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(data[:,0], data[:,1])
ax.axhline(0)
ax.axvline(0)
fig.savefig("test-output.png")


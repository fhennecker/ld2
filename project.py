import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

np.random.seed(42)
size = 600
# grid = np.random.randint(2, size=(size,size))
grid = np.random.rand(size, size) > 0.5
# plt.imshow(grid, interpolation="nearest")
# plt.show()

def m(v):
    """Moore kernel"""
    return np.array([[v]*3, [v, 0, v], [v]*3])

def payoffs(grid):
    defecter_payoff = convolve2d(grid, m(10), mode='same', boundary='wrap') * (1 - grid) 
    cooperators_payoff = convolve2d(grid, m(7), mode='same', boundary='wrap') * grid
    return defecter_payoff + cooperators_payoff


def new_state(grid, payoffs):
    new_grid = np.zeros_like(grid)
    size = len(grid)
    for row in range(size):
        for col in range(size):
            max_index = None
            max_value = -1
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    index = ((row+dr)%size, (col+dc)%size)
                    value = payoffs[index] 
                    if max_index is None or value > max_value:
                        max_index, max_value = index, value
            new_grid[row, col] = grid[max_index]
    return new_grid
                        
print(payoffs(grid))
coop = []
from time import time
start = time()
for i in range(100):
    grid = new_state(grid, payoffs(grid))
    coop.append(grid.mean())
print(time()-start)

plt.imshow(grid, interpolation='nearest')
plt.show()

plt.plot(coop)
plt.show()

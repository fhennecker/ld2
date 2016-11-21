import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

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
                        
def plot_n_runs(n_simulations, n_steps, size):
    coop = np.zeros((n_simulations, n_steps))

    for s in range(n_simulations):
        print s,
        grid = np.random.randint(2, size=(size,size))
        for i in range(n_steps):
            coop[s,i] += grid.mean()
            grid = new_state(grid, payoffs(grid))

    plt.figure(1, figsize=(10,4))
    axlines = plt.axes([0.06,0.15, 0.58, 0.85])
    axhist = plt.axes([0.67,0.15, 0.32,0.85])
    axhist.yaxis.set_major_formatter(plt.NullFormatter())
    axhist.set_xlabel("Final cooperation distribution")
    axlines.set_xlabel("Time")
    axlines.set_ylabel("Cooperation")

    for s in range(n_simulations):
        axlines.plot(coop[s,:], 'b-', alpha=0.1)
    axlines.plot(np.mean(coop, 0), 'r', linewidth=2)

    import matplotlib.lines as mlines
    axlines.legend(loc=4, handles=[mlines.Line2D([],[], color='red',
        linewidth=2, label='Average')])
    axlines.set_ylim([-0.1, 1.1])
    axhist.set_ylim([-0.1, 1.1])
    axhist.hist(coop[:,-1], orientation='horizontal', bins=np.arange(0,1.1,0.1))
    plt.show()

def show_at_steps(size, steps):
    grid = np.random.randint(2, size=(size, size))
    for i in range(max(steps)+1):
        if i in steps:
            plt.imshow(grid, interpolation='NEAREST', cmap='cool')
            plt.show()
        grid = new_state(grid, payoffs(grid))

if __name__ == '__main__':
    # plot_n_runs(100, 60, 50)
    show_at_steps(12, [0, 1, 5, 10, 20, 50])

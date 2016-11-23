import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from random import shuffle, choice, random

def moore(v):
    """Moore kernel"""
    return np.array([[v]*3, [v, 0, v], [v]*3])

def vonnneumann(v):
    """Moore kernel"""
    return np.array([[0,v,0], [v,0,v], [0,v,0]])

def payoffs(grid, method='moore'):
    f = moore
    if method == 'vn':
        f = vonnneumann
    defecter_payoff = convolve2d(grid, f(10), mode='same', boundary='wrap') * (1 - grid) 
    cooperators_payoff = convolve2d(grid, f(7), mode='same', boundary='wrap') * grid
    return defecter_payoff + cooperators_payoff

def snowdrift_payoffs(grid, method='moore'):
    f = moore
    if method == 'vn':
        f = vonnneumann
    defecter_payoff = convolve2d(grid, f(10), mode='same', boundary='wrap') * (1 - grid) 
    cooperators_payoff = convolve2d(grid, f(7), mode='same', boundary='wrap') * grid
    cooperators_payoff += convolve2d((1-grid), f(3), mode='same', boundary='wrap') * grid
    return defecter_payoff + cooperators_payoff

def new_state(grid, payoffs, method='moore'):
    new_grid = np.zeros_like(grid)
    size = len(grid)
    for row in range(size):
        for col in range(size):
            max_index = None
            max_value = -1
            if method == 'moore':
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        index = ((row+dr)%size, (col+dc)%size)
                        value = payoffs[index] 
                        if max_index is None or value > max_value:
                            max_index, max_value = index, value
            elif method == 'vn':
                choices = [[-1, 0], [1, 0], [0, -1], [0, 1], [0,0]]
                shuffle(choices)
                for dr, dc in choices:
                    index = ((row+dr)%size, (col+dc)%size)
                    value = payoffs[index]
                    if max_index is None or value > max_value:
                        max_index, max_value = index, value
            new_grid[row, col] = grid[max_index]
    return new_grid
                        
def snowdrift_new_state(grid, payoffs, method='moore'):
    new_grid = np.zeros_like(grid)
    size = len(grid)
    neighbours = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    if method == 'moore':
        neighbours += [[-1, 1], [-1, -1], [1, -1], [1, 1]]
    for row in range(size):
        for col in range(size):
            c = choice(neighbours)
            neighbour = ((row+c[0])%size, (col+c[1])%size)
            N = 8 if method == 'moore' else 4
            p = 1.*(1 + 1.*(payoffs[neighbour]-payoffs[row, col])/(N*10))/2
            if random() <= p:
                new_grid[row, col] = grid[neighbour]
            else:
                new_grid[row, col] = grid[row, col]
    return new_grid

def plot_n_runs(n_simulations, n_steps, size, method, game):
    coop = np.zeros((n_simulations, n_steps))

    for s in range(n_simulations):
        print s
        grid = np.random.randint(2, size=(size,size))
        for i in range(n_steps):
            coop[s,i] += grid.mean()
            if game == 'prisoner':
                grid = new_state(grid, payoffs(grid, method), method)
            elif game == 'snowdrift':
                grid = snowdrift_new_state(grid, snowdrift_payoffs(grid, method), method)

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

def show_at_steps(size, steps, method, game):
    grid = np.random.randint(2, size=(size, size))
    for i in range(max(steps)+1):
        if i in steps:
            plt.imshow(grid, interpolation='NEAREST', cmap='cool')
            #plt.show()
            plt.waitforbuttonpress()
        if game == 'prisoner':
            grid = new_state(grid, payoffs(grid, method), method)
        elif game == 'snowdrift':
            grid = snowdrift_new_state(grid, snowdrift_payoffs(grid, method), method)


if __name__ == '__main__':
    #  plot_n_runs(100, 60, 50, "vn", 'snowdrift')
    #  show_at_steps(50, range(20), 'moore', 'snowdrift')

    grid = np.zeros((50,50))
    grid[24:27,24:27] = 1
    for i in range(50):
        plt.imshow(grid, interpolation='NEAREST', cmap='cool')
        plt.waitforbuttonpress()
        grid = new_state(grid, payoffs(grid, 'vn'), 'vn')

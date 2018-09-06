import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.table import Table

World_size = 5
Discount = 0.9

A_pos = [0, 1]
A_pos_change = [4, 1]
B_pos = [0, 3]
B_pos_change = [2, 3]

Actions = [np.array([-1, 0]),       # 向北
           np.array([1, 0]),        # 向南
           np.array([0, -1]),       # 向西
           np.array([0, 1])]        # 向东
Action_Prob = 0.25

def step(state, action):
    if state == A_pos:
        return A_pos_change, 10
    elif state == B_pos:
        return B_pos_change, 5
    state = np.array(state)
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= World_size or y < 0 or y >= World_size:
        next_state = state
        reward = -1
    else:
        reward = 0
    return next_state, reward

def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox = [0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for(i, j), val in np.ndenumerate(image):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        color = 'white'

        tb.add_cell(i, j, width, height, text = val, loc = 'center', facecolor = color)
        # Row Labels...
        tb.add_cell(i, -1, width, height, text = i+1, loc = 'right', edgecolor = 'none', facecolor = 'none')
        # Column Labels...
        tb.add_cell(-1, j, width, height/2, text = j+1, loc = 'center', edgecolor = 'none', facecolor = 'none')

        ax.add_table(tb)

def figure_1():
    value = np.zeros((World_size, World_size))
    while True:
        # keep iteration until convergence
        new_value = np.zeros(value.shape)
        for i in range(0, World_size):
            for j in range(0, World_size):
                for action in Actions:
                    (next_i, next_j), reward = step([i,j], action)
                    new_value[i,j] += Action_Prob * (reward + Discount * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            break
        value = new_value
    draw_image(np.round(new_value, decimals = 2))
    plt.savefig('images/figure_1.png')
    plt.close()

def figure_2():
    value = np.zeros((World_size, World_size))
    while True:
        # keep iteration until convergence
        new_value = np.zeros(value.shape)
        for i in range(0, World_size):
            for j in range(0, World_size):
                values = []
                for action in Actions:
                    (next_i, next_j), reward = step([i, j], action)
                    values.append(reward + Discount * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(value - new_value)) < 1e-4:
            break
        value = new_value
    draw_image(np.round(new_value, decimals = 2))
    plt.savefig('images/figure_2.png')
    plt.close()

if __name__ == '__main__':
    figure_1()
    figure_2()

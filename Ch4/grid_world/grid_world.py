import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.table import Table

World_size = 4
Terminal_state = [(0,0), (3,3)]
Actions = [np.array([-1, 0]),
          np.array([0, -1]),
          np.array([1, 0]),
          np.array([0, 1])]
Action_prob = 0.25
Reward_unterminal = -1
Reward_terminal = 0

def step(state, action):
    if state in Terminal_state:
        return state, Reward_terminal
    state = np.array(state)
    new_state = (state + action).tolist()
    x, y = new_state
    if x < 0 or x >= World_size or y < 0 or y >= World_size:
        new_state = state
    return new_state, Reward_unterminal

def draw_fig( datas ):
    fig, ax = plt.subplots( )
    ax.set_axis_off( )
    tb = Table( ax, bbox = [0, 0, 1, 1] )

    nrows, ncols = datas.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate( datas ):
        # Index either the first or second item of bkg_colors based on
        # a checker board pattern
        color = 'white'

        tb.add_cell( i, j, width, height, text = val, loc = 'center', facecolor = color )
        # Row Labels...
        tb.add_cell( i, -1, width, height, text = i + 1, loc = 'right', edgecolor = 'none', facecolor = 'none' )
        # Column Labels...
        tb.add_cell( -1, j, width, height / 2, text = j + 1, loc = 'center', edgecolor = 'none', facecolor = 'none' )

        ax.add_table( tb )

def prediction( iter_num ):
    values = np.zeros((World_size, World_size))
    for k in range(iter_num + 1):
        new_values = np.zeros((World_size, World_size))
        for i in range(World_size):
            for j in range(World_size):
                if i == j == 0 or i == j == 3:
                    continue
                else:
                    for action in Actions:
                        (next_state_i, next_state_j), reward = step([i, j], action)
                        new_values[i, j] += Action_prob * (reward + values[next_state_i, next_state_j])
        values = new_values
    draw_fig(np.round(values, decimals = 2))
    plt.savefig('images/figure.png')
    plt.close()

if __name__ == '__main__':
    prediction(100)




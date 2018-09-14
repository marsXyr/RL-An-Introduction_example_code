import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import exp, factorial
import seaborn as sns

# maximum # of cars in each location
MAX_CARS = 20

# maximum # of cars to move during night
MAX_MOVE_OF_CARS = 5

# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3

# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4

# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3

# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2

DISCOUNT = 0.9

# credit earned by a car
RENTAL_CREDIT = 10

# cost of moving a cars
MOVE_CAR_COST = 2

# cost of # of cars more than 10
MORE_THAN_10 = 4

# all possible actions
actions = np.arange(-MAX_MOVE_OF_CARS, MAX_MOVE_OF_CARS + 1)

# An upper bound for possion distridution
# If n is greater than this value, then the probability of getting n is truncated to 0
POSSION_UPPER_BOUND = 11

# Probability for possion distribution
# @lam: lambda should be less than 10 for this function
possion_cache = dict()

def possion(n, lam):
    global possion_cache
    key = n * 10 + lam
    if key not in possion_cache.keys():
        possion_cache[key] = exp(-lam) * pow(lam, n) / factorial(n)
    return possion_cache[key]


def expected_return(state, action, state_value, constant_returned_cars ):
    # initailize total return
    returns = 0.0

    # cost for moving cars
    returns -= MOVE_CAR_COST * abs(action)

    # go through all possible rental requests
    for rental_request_first_loc in range(0, POSSION_UPPER_BOUND):
        for rental_request_second_loc in range(0, POSSION_UPPER_BOUND):
            # moving cars
            num_of_cars_first_loc = int(min(state[0] - action, MAX_CARS))
            num_of_cars_second_loc = int(min(state[1] + action, MAX_CARS))
            # valid rental requests should be less than actual # of cars
            real_rental_first_loc = min(num_of_cars_first_loc, rental_request_first_loc)
            real_rental_second_loc = min(num_of_cars_second_loc, rental_request_second_loc)
            # get credits for renting
            reward = (real_rental_first_loc + real_rental_second_loc) * RENTAL_CREDIT
            num_of_cars_first_loc -= real_rental_first_loc
            num_of_cars_second_loc -= real_rental_second_loc
            # probability for current combination of rental requests
            prob = possion(rental_request_first_loc, RENTAL_REQUEST_FIRST_LOC) * possion(rental_request_second_loc, RENTAL_REQUEST_SECOND_LOC)

            more_cost = 0
            if constant_returned_cars:
                # get returned cars, those cars can be used for renting tomorrow
                returnd_cars_first_loc = RETURNS_FIRST_LOC
                returnd_cars_second_loc = RETURNS_SECOND_LOC
                num_of_cars_first_loc = min(num_of_cars_first_loc + returnd_cars_first_loc, MAX_CARS)
                num_of_cars_second_loc = min(num_of_cars_second_loc + returnd_cars_second_loc, MAX_CARS)
                if num_of_cars_first_loc >= 10:
                    more_cost += 4
                if num_of_cars_second_loc >= 10:
                    more_cost += 4
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc]) + more_cost
            else:
                for returned_cars_first_loc in range(0, POSSION_UPPER_BOUND):
                    for returned_cars_second_loc in range(0, POSSION_UPPER_BOUND):
                        num_of_cars_first_loc_ = min(num_of_cars_first_loc + returnd_cars_first_loc, MAX_CARS)
                        num_of_cars_second_loc_ = min(num_of_cars_second_loc + returnd_cars_second_loc, MAX_CARS)
                        if num_of_cars_first_loc >= 10:
                            more_cost += 4
                        if num_of_cars_second_loc >= 10:
                            more_cost += 4
                        prob_ = possion(returnd_cars_first_loc, RETURNS_FIRST_LOC) * possion(returnd_cars_second_loc, RETURNS_SECOND_LOC) * prob
                        returns += prob_ * (reward + DISCOUNT * state_value[num_of_cars_first_loc_, num_of_cars_second_loc_]) + more_cost
    return returns

def figure_1(constant_returned_cars = True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype = np.int)
    iterations = 0
    _, axes = plt.subplots(2, 3, figsize = (40, 20))
    plt.subplots_adjust(wspace = 0.1, hspace = 0.2)
    axes = axes.flatten()
    while True:
        fig = sns.heatmap(np.flipud(policy), cmap = 'YlGnBu', ax = axes[iterations])
        fig.set_title('policy %d' % (iterations), fontsize = 30)
        fig.set_xlabel('# cars at second location', fontsize = 30)
        fig.set_ylabel('# cars at first location', fontsize = 30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))

        # policy evaluation (in-place)
        while True:
            new_value = np.copy(value)
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    new_value[i,j] = expected_return([i,j], policy[i,j], new_value, constant_returned_cars)
            value_change = np.abs((new_value - value)).sum()
            print('value change %f' % (value_change))
            value = new_value
            if value_change < 1e-4:
                break

        # policy improvement
        new_policy = np.copy(policy)
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                action_returns = []
                for action in actions:
                    if (action >= 0 and action <= i) or (action < 0 and abs(action) <= j):
                        action_returns.append(expected_return([i,j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-float('inf'))
                new_policy[i,j] = actions[np.argmax(action_returns)]

        policy_change = (new_policy != policy).sum()
        print('policy changed in %d states' %(policy_change))
        policy = new_policy
        if policy_change == 0:
            fig = sns.heatmap( np.flipud( value ), cmap = 'YlGnBu', ax = axes[-1] )
            fig.set_title( 'optimal value', fontsize = 30 )
            fig.set_xlabel( '# cars at second location', fontsize = 30 )
            fig.set_ylabel( '# cars at first location', fontsize = 30 )
            fig.set_yticks( list( reversed( range( MAX_CARS + 1 ) ) ) )
            break

        iterations += 1

    plt.savefig('images/figure_1.png')

if __name__ == '__main__':
    figure_1( )
import sys
from collections import defaultdict

import matplotlib
import numpy as np
import random as rd

if "../" not in sys.path:
    sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

# create env
env = BlackjackEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn
'''
def strategy(observation):
    score, dealer_score, usable_ace = observation
    # Stick (action 0) if the score is > 20, hit (action 1) otherwise
    return 0 if score >= 20 else 1
'''
def strategy(actions_prob):
    p = rd.random()
    best_action = np.argmax(actions_prob)
    action_ = np.argmin(actions_prob)
    return best_action if p <= actions_prob[best_action] else action_

def mc(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).

    ##***初始化Q的key value为-1的矩阵***##
    Q = defaultdict(lambda: np.zeros(env.action_space.n) - np.ones(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

    #############################################Implement your code###################################################################################################
    # step 1 : Generate an episode.
    # An episode is an array of (state, action, reward) tuples
        episode = []
        observation = env.reset()
        for t in range(100):
            action = strategy(policy(observation))
            observation_, reward, done, _ = env.step(action)
            episode.append([observation, action, reward])
            if done:
                break
            observation = observation_

    # step 2 : Find all (state, action) pairs we've visited in this episode
        G = 0
        for t in range(len(episode)-1, -1, -1):
            state = episode[t][0]; action = episode[t][1]; reward = episode[t][2]
            G += reward

    # step 3 : Calculate average return for this state over all sampled episodes
            if (state, action) not in returns_sum:
                returns_sum[(state, action)] = G
                returns_count[(state, action)] = 1
            else:
                returns_count[(state, action)] += 1
                returns_sum[(state, action)] += G
            Q[state][action] += 1 / returns_count[(state, action)] * (G - Q[state][action])
#           Q[state][action] = returns_sum[(state, action)] / returns_count[((state, action))]
            policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    #############################################Implement your code end###################################################################################################

    return Q, policy


Q, policy = mc(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
print (V)
plotting.plot_value_function(V, title="Optimal Value Function")
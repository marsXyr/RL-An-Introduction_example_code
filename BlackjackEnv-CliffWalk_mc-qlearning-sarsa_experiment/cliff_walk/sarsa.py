import gym
import itertools
import matplotlib
import numpy as np
import pandas as pd
import sys
import random as rd

if "../" not in sys.path:
    sys.path.append("../")

from collections import defaultdict
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()


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

def strategy(actions_prob):
    p = rd.random()
    actions = [0, 1, 2, 3]
    best_action = np.argmax(actions_prob)
    if p <= np.max(actions_prob):
        return best_action
    else:
        return rd.choice(actions)

def sarsa(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1):
    """
    SARSA algorithm: On-policy TD control. Finds the optimal epsilon-greedy policy.

    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, stats).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        # Reset the environment
        state = env.reset()
        action = strategy(policy(state))
        # One step in the environment
        for t in itertools.count():
            #########################################Implement your code here#######################################################################################
            # step 1 : Take a step( 1 line code, tips : env.step() )
            state_, reward, done, _ = env.step(action)

            # step 2 : Pick the next action

            action_ = strategy(policy(state_))

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # step 3 : TD Update
            # compute Q value
            Q[state][action] += alpha * ( reward + discount_factor * Q[state_][action_] - Q[state][action] )
            state = state_; action = action_
            if done:
                break

    #########################################Implement your code here end#####################################################################################
    return Q, stats


Q, stats = sarsa(env, 500)

plotting.plot_episode_stats(stats)
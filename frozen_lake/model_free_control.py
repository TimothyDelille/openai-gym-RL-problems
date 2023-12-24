from typing import DefaultDict
import numpy as np
from pprint import pprint
from itertools import combinations_with_replacement
import gym
from tqdm import tqdm
from collections import defaultdict

def monte_carlo_policy_evaluation(env, policy, episodes, gamma=1., first_visit_only=False):
    n_visits = np.zeros(env.env.nS)
    V = np.zeros(env.env.nS)

    for history in episodes:
        G = sum([gamma**i*r for i, (_, _, r) in enumerate(history)])
        for s, a, r in history:
            if first_visit_only and n_visits[s] >= 1:
                continue

            n_visits[s] += 1

            V[s] = V[s] + 1/n_visits[s]*(G - V[s])
            G = (G - r)/gamma
    # V[env.env.nS - 1] = 1
    return V

def record_history(env, policy, s=0):
    """
    policy has shape nS, nA and contains probability of taking action a given state s
    """
    history = []
    done = False
    env.reset()  # very important. resets the episode.
    while not done:
        a = np.random.choice(range(env.env.nA), p=policy[s])
        s_, r, done, _ = env.step(a)
        history.append((s, a, r))
        s = s_  # also very important and easily forgettable.
    return history


def importance_sampling(p, q, episodes, gamma=1.):
    """
    Args:
        p (np.array with shape (env.env.nS, env.env.nA)): the policy under which the samples were made
        q (np.array with shape (env.env.nS, env.env.nA)): the policy we want to evaluate
        episodes: List[Dict[str: List]] each dictionary has keys action, reward, state
        gamma (float): discount factor
    """

    V = 0
    for history in episodes:
        prod = 1
        G = 0
        k = 0
        for s, a, r in history:
            G += gamma**k*r
            prod *= q[s, a]/p[s, a]
        V += G*prod
    V = V/len(episodes)
    return V

def pprint_policy(env, policy):
    action_to_arrow = {0: "<", 1: "v", 2: ">", 3: "^"}
    pretty_policy = np.array([action_to_arrow[x] for x in policy]).reshape(env.env.nrow, env.env.ncol)
    pprint(pretty_policy)

def pprint_value(env, V):
    pprint(V.reshape(env.env.nrow, env.env.ncol))

def policy_evaluation(env, policy):
    episodes = [record_history(env, np.random.randint(0, env.env.nA, size=env.env.nS, dtype=np.int8)) for _ in range(1000)]
    # episodes = [record_history(env, policy) for _ in range(1000)]
    p = 1/env.env.nA*np.ones((env.env.nS, env.env.nA))
    q = np.zeros((env.env.nS, env.env.nA))
    q[np.arange(env.env.nS), policy] = 1.

    # return monte_carlo_policy_evaluation(env, policy, episodes)
    return importance_sampling(p, q, episodes)

def td_learning(env, policy, episodes, alpha, gamma):
    V = np.zeros(env.env.nS)
    for history in episodes:
        for i in range(len(history)-1):
            s,a,r = history[i]
            s_, a_, r_ = history[i+1]

            V[s] = V[s] + alpha*(r + gamma*V[s_] - V[s])
    return V

def policy_search(env):
    #policies = combinations_with_replacement(range(env.env.nA), env.env.nS)

    policies = [np.random.randint(0, env.env.nA, size=env.env.nS, dtype=np.int8) for _ in range(1000)]
    best_V = np.zeros(env.env.nS)
    best_policy = np.random.randint(0, env.env.nA, size=env.env.nS)

    for policy in tqdm(policies, total=len(policies)):
        V = policy_evaluation(env, policy)
        if np.all(V > best_V):
            best_policy = np.array(policy)
            best_V = V.copy()
    return best_V, best_policy


def epsilon_greedy(env, epsilon, Q):
    policy = epsilon/(env.env.nA-1)*np.ones((env.env.nS, env.env.nA))
    policy[np.arange(env.env.nS), np.argmax(Q, axis=1)] = 1 - epsilon

    policy[Q.max(-1) == Q.min(-1), :] = 1/env.env.nA
    return policy


def random_uniform_policy(env):
    return 1/env.env.nA*np.ones((env.env.nS, env.env.nA))
def online_monte_carlo_control(env, iterations=10000):
    policy = random_uniform_policy(env)

    Q = np.zeros((env.env.nS, env.env.nA))
    epsilon = 1
    n_visits = np.zeros((env.env.nS, env.env.nA))

    for k in range(2, iterations + 2):
        history = record_history(env, policy)
        sum_of_returns = sum(r for _, _, r in history)
        for s,a,r in history:
            n_visits[s, a] += 1
            Q[s, a] = (Q[s, a]*(n_visits[s, a] - 1) + sum_of_returns)/n_visits[s, a]
            sum_of_returns -= r
        epsilon = 1/k
        policy = epsilon_greedy(env, epsilon, Q)
    return Q, policy


def sarsa(env, iterations=10000, alpha=0.1, gamma=1, epsilon=0.1):
    Q = np.zeros((env.env.nS, env.env.nA))
    policy = epsilon_greedy(env, epsilon, Q)
    for _ in range(iterations):
        history = record_history(env, policy)
        for i in range(1, len(history)):
            s, a, r = history[i-1]
            s_, a_, r_ = history[i]
            Q[s, a] = Q[s, a] + alpha*(r + gamma*Q[s_, a_] - Q[s, a])
        Q[s_, a_] = Q[s_, a_] + alpha*(r_ - Q[s_, a_])  # Q is set to zero for terminal states
        policy = epsilon_greedy(env, epsilon, Q)
    return Q, policy

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')
    env.reset()
    #policy = [0.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0]
    Q, policy = sarsa(env)
    V = np.sum(policy*Q, axis=1)
    print(policy.argmax(1))
    pprint_policy(env, policy.argmax(axis=1))
    #print(Q)
    true_V = np.array([[0.82, 0.82, 0.82, 0.82],
                       [0.82, 0., 0.53, 0.],
                       [0.82, 0.82, 0.76, 0.],
                       [0., 0.88, 0.94, 0.]])
    print(true_V)
    print(V.reshape(4,4))
    print(true_V - V.reshape(4,4))
    env.close()

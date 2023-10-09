import gym
import numpy as np
from pprint import pprint

from policy_iteration import policy_iteration
from value_iteration import value_iteration

# SFFF       (S: starting point, safe)
# FHFH       (F: frozen surface, safe)
# FFFH       (H: hole, fall to your doom)
# HFFG       (G: goal, where the frisbee is located)

def main():
    env = gym.make('FrozenLake-v1')
    env.reset()
    action_to_arrow = {0: "<", 1: "v", 2: ">", 3: "^"}
    # print(env.env.desc)
    print("\nPOLICY ITERATION\n")
    V, policy = policy_iteration(env)
    V = V.reshape(env.env.nrow, env.env.ncol)
    policy_repr = np.array([action_to_arrow[x] for x in policy]).reshape(env.env.nrow, env.env.ncol)
    pprint(V)
    pprint(policy_repr)

    print("\nVALUE ITERATION\n")
    V, policy = value_iteration(env)
    V = V.reshape(env.env.nrow, env.env.ncol)
    policy_repr = np.array([action_to_arrow[x] for x in policy]).reshape(env.env.nrow, env.env.ncol)
    pprint(V)
    pprint(policy_repr)
    return V, policy

if __name__ == "__main__":
    main()

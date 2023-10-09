import numpy as np

def policy_evaluation(env, policy, gamma=1., epsilon=10e-9):
    # policy = list of length nb_states and containing an integer between 0 and nb_actions - 1
    new_V = np.zeros(env.env.nS)
    V = 1e9*np.ones(env.env.nS)
    while np.abs(V - new_V).max() > epsilon:
        V = new_V.copy()
        for s in range(env.env.nS):
            new_V[s] = sum(p_*(r_ + gamma*V[s_]) for p_, s_, r_, _ in env.env.P[s][policy[s]])
    return new_V


def policy_improvement(env, V, gamma=1.):
    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        policy[s] = np.argmax([
            sum(p_*(r_ + gamma*V[s_]) for p_, s_, r_, _ in env.env.P[s][a]) for a in range(env.env.nA)
            ])
    return policy

def policy_iteration(env, gamma=1.):
    policy = np.random.randint(0, env.env.nA, size=env.env.nS)

    while True:
        V = policy_evaluation(env, policy, gamma)
        new_policy = policy_improvement(env, V, gamma)

        if (new_policy == policy).all():
            break
        policy = new_policy.copy()
    return V, policy

import numpy as np

def value_iteration(env, gamma=1., epsilon=10e-9):
    new_V = np.zeros(env.env.nS)
    V = 1e9*np.ones(env.env.nS)
    while np.abs(V - new_V).max() > epsilon:
        V = new_V.copy()
        for s in range(env.env.nS):
            new_V[s] = max([
                sum(p*(r + gamma*V[s_]) for p, s_, r, _ in env.env.P[s][a])
            for a in range(env.env.nA)
            ])

    policy = np.zeros(env.env.nS)
    for s in range(env.env.nS):
        policy[s] = np.argmax([
                sum(p_*(r_ + gamma*V[s_]) for p_, s_, r_, _ in env.env.P[s][a]) for a in range(env.env.nA)
        ])
    return V, policy

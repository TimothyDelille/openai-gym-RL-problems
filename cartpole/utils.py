from collections import namedtuple
import numpy as np

import imageio
import cv2

# Record namedtuple contains action, observation, reward and frame for a single time step.
# An episode is a collection of records.
Record = namedtuple("Record", "action observation reward frame")

# sample_episode is an agnostic function used to sample an episode using a policy. 
# The policy must take an observation and return an action.
# The function returns an episode as a list of `Record` tuples.
# Doesn't do any update during the episode.
# render (bool) determines whether to render frames.
def sample_episode(policy, env, render=False):
    episode = []
    terminated = False
    truncated = False
    step = 0
    obs, info = env.reset()
    while not (terminated or truncated):
        step += 1
        frame = None
        if render:
            frame = env.render()
            cv2.putText(
                img=frame, 
                text=f'steps={step}/500', 
                org=(10, 50), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=1, 
                color=(0, 0, 255), 
                thickness=2, 
                lineType=cv2.LINE_AA
            )
        action = policy(obs)
        new_obs, reward, terminated, truncated, info = env.step(action)
        episode.append(
            Record(
                action=action, 
                observation=obs, 
                reward=reward,
                frame=frame,
            ),
        )
        obs = new_obs

    if render:
        fps = 30
        imageio.mimsave("cartpole.gif", [e.frame for e in episode], duration=len(episode)/fps)
    return episode

# digitize quantizes the observations into bins. Returns bin indexes (0-based).
def digitize(obs, bins):
    res = []
    for o, b in zip(obs, bins):
        res.append(
            np.digitize(o, b) - 1
        )
    return res

# define bins for the continuous observations.
def get_bins(bin_size):
    return [
        np.linspace(-4.8, 4.8, bin_size),
        np.linspace(-4, 4, bin_size),
        np.linspace(-0.418, 0.418, bin_size),
        np.linspace(-4, 4, bin_size)
    ]
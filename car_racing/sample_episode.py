from collections import deque
import os

import imageio
import gymnasium as gym
from tqdm import tqdm
import torch
import numpy as np

from deep_q_network import DQN
from policy_gradient import Policy

@torch.no_grad()
def sample_episode(ckpt_path):
    frame_stack_len = 4
    # model = DQN(frame_stack_len=frame_stack_len)
    model = Policy(frame_stack_len=frame_stack_len)
    model.load_state_dict(torch.load(ckpt_path))
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=True, render_mode="rgb_array")
    frames = []
    done = False
    obs, info = env.reset()
    max_frames = 600
    counter = 1
    s = model.preprocess(obs)
    frame_stack = deque(maxlen=frame_stack_len)
    frame_stack.extend([s for _ in range(frame_stack_len)])
    pbar = tqdm(total=max_frames)
    while not done and counter <= max_frames:
        frame = env.render()
        frames.append(frame)
        state = torch.concatenate(list(frame_stack), dim=0)
        state = state.unsqueeze(0)
        # code for deep q network:
        # a = model.act(epsilon=0, state=state)
        # action = model.action_space[a]
        # code for policy gradient:
        mean, cov, v = model.forward(state)
        eps = np.random.normal(0, 1, size=model.n_actions)
        action = mean + torch.sqrt(cov) * torch.tensor(eps, requires_grad=False)
        action = action.flatten()
        action = action.detach().numpy()
        new_obs, reward, terminated, truncated, info = env.step(action)
        frame_stack.append(model.preprocess(new_obs))
        done = terminated or truncated
        counter += 1
        pbar.update(1)
    pbar.close()
    env.close()

    print("💾 Saving to GIF...")
    fps = 60
    imageio.mimsave(os.path.join(os.path.dirname(__file__), "carracing.gif"), frames, duration=len(frames)/fps)
    print("🚀 Done!")
    return
    
if __name__ == "__main__":
    sample_episode(
        ckpt_path=os.path.join(
            os.path.dirname(__file__),
            "checkpoints/policy_gradient/600.pt"
        ),
    )           
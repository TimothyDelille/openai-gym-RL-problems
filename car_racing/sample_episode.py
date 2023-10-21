from collections import deque

import imageio
import gymnasium as gym
from tqdm import tqdm
import torch

from model import DQN

def sample_episode():
    frame_stack_len = 3
    model = DQN(frame_stack_len=frame_stack_len)
    model.load_state_dict(torch.load("../checkpoints/1200.pt"))
    env = gym.make("CarRacing-v2", domain_randomize=False, continuous=True, render_mode="rgb_array")
    frames = []
    done = False
    obs, info = env.reset()
    max_frames = 600
    counter = 1
    s = model.transforms(obs)
    frame_stack = deque(maxlen=frame_stack_len)
    frame_stack.extend([s for _ in range(frame_stack_len)])
    pbar = tqdm(total=max_frames)
    while not done and counter <= max_frames:
        frame = env.render()
        frames.append(frame)
        state = torch.concatenate(list(frame_stack), dim=0)
        state = state.unsqueeze(0)
        a = model.act(epsilon=0, state=state)
        action = model.action_space[a]
        new_obs, reward, terminated, truncated, info = env.step(action)
        frame_stack.append(model.transforms(new_obs))
        done = terminated or truncated
        counter += 1
        pbar.update(1)
    env.close()

    print("💾 Saving to GIF...")
    fps = 60
    imageio.mimsave('carracing.gif', frames, duration=len(frames)/fps)
    print("🚀 Done!")
    return
    
if __name__ == "__main__":
    sample_episode()                
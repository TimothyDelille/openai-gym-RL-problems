import logging
from collections import deque
import numpy as np
import os

import torch
import gymnasium as gym
from tqdm.auto import tqdm
# local imports.
from metrics import Metrics
from deep_q_network import DQN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            logger.info("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            logger.info("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        device = torch.device("cpu")
    else:
        device = torch.device("mps")
    logger.info(f"Using device: {device}")
    return device

CHECKPOINT_PATH = "./checkpoints"
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)
    
# state size = 16*16 img *4 frames = 1024 bytes
# 100 MB buffer size = 100 000 states
# buffer has to be large enough to break correlations between multiple states
BUFFER_SIZE = 100 # int(10000)  # 1e6 in the paper

BATCH_SIZE = 32
# reward discounting value
GAMMA = 0.95

# number of frames that the model processes at once
FRAME_STACK_LEN = 3
# number of frames where we will act the previous action.
SKIP_FRAMES = 4
LEARNING_RATE = 1e-4


# atari paper
# "epsilon annealed linearly from 1 to 0.1 over the first million
# frames, and fixed at 0.1 thereafter"
EPSILON_START = 1
EPSILON_END = 0.1
def get_epsilon(epsilon_start: float, epsilon_end: float, frame: int):
    n_frames = 1e6
    global epsilon
    if frame >= n_frames:
        return epsilon_end
    else:
        return (epsilon_end - epsilon_start) / (n_frames - 1) * (frame - 1) + epsilon_start

def sync_models(model, target_model, device, path: str):
    torch.save(model.state_dict(), path)
    target_model.load_state_dict(torch.load(path, map_location=device))

    
def main():
    device = get_device()
    model = DQN(frame_stack_len=FRAME_STACK_LEN, learning_rate=LEARNING_RATE)
    model.to(device)
    # target_model = DQN()
    # target_model.to(device)
    # sync_models(model, target_model)

    metrics = Metrics(tensorboard=True)
    # domain_randomize: background and track colours are different on every reset.
    env = gym.make("CarRacing-v3", domain_randomize=False, continuous=True)
    buffer = deque(maxlen=BUFFER_SIZE)  # will automatically pop items when we go over the buffer_size
    state_shape = [FRAME_STACK_LEN, 84, 84]
    state_size = np.prod(state_shape)
    buffer = torch.zeros((BUFFER_SIZE, 2 * state_size + 3), device="cpu")
    buffer_index = 0
    buffer_full = False

    n_updates = int(10e6)  # 10M frames in the paper
    pbar = tqdm(total=n_updates)
    updates_counter = 0
    episode_counter = 0
    best_total_reward = 0
    sync_models_frequency = 5  # episodes

    

    while updates_counter < n_updates:
        episode_counter += 1
        # normal reset changes the colour scheme by default
        obs, info = env.reset()
        s = model.preprocess(obs)
        frame_stack = deque(maxlen=FRAME_STACK_LEN)
        frame_stack.extend([s for _ in range(FRAME_STACK_LEN)])  # init with the same state.
        done = False
        total_loss = 0
        total_episode_reward = 0
        episode_length = 0
        # Sample 1 episode.
        while not done:
            episode_length += 1
            state = torch.concatenate(list(frame_stack), dim=0)
            state = state.unsqueeze(0)  # add batch dim
            epsilon = get_epsilon(epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, frame=updates_counter)
            a = model.act(epsilon, state.to(device))  # action index. we will store it in the buffer.
            a_arr = model.action_space[a]  # tuple (steering, gas, break)
            reward = 0
            # atari paper: we fixed all positive rewards to be 1 and all negative rewards to be −1,
            # leaving 0 rewards unchanged.
            scaled_reward = 0
            # repeat the same action over skip_frames
            for _ in range(SKIP_FRAMES):
                new_obs, r, terminated, truncated, info = env.step(a_arr)
                reward += r
                if r != 0:
                    scaled_reward += 1.0 if r > 0 else -1.0

                new_s = model.preprocess(new_obs)
                frame_stack.append(new_s)

                # Termination refers to the episode ending after reaching a terminal state that is defined as part of the environment definition.
                # Truncation refers to the episode ending after an externally defined condition (that is outside the scope of the Markov Decision Process). This could be a time-limit, a robot going out of bounds etc.
                done = terminated or truncated
                if done:
                    break
            
            if scaled_reward != 0:
                scaled_reward = 1.0 if scaled_reward > 0 else -1.0

            total_episode_reward += reward
            
            new_state = torch.concatenate(list(frame_stack), dim=0)
            # buffer.append((state, a, scaled_reward, new_state, done))
            buffer[(buffer_index + 1) % BUFFER_SIZE] = torch.cat(
                [
                    state.reshape(-1),
                    new_state.reshape(-1),
                    torch.tensor([a]),
                    torch.tensor([scaled_reward]),
                    torch.tensor([1.0] if done else [0.0]),
                ]
            , axis=0)
            if buffer_index == BUFFER_SIZE - 1:
                buffer_full = True
            buffer_index = (buffer_index + 1) % BUFFER_SIZE
            # no need to wait for full buffer to start training. 

            # early termination of negative episodes. 
            # This helps training, probably because the agent can play more episodes this way.
            # if done or negative_reward_counter >= 25 or total_reward < 0:
            #     break

            # construct batch.
            # batch = {"r": [], "done": [], "s": [], "new_s": [], "a": []}
            sampled_indices = np.random.choice(np.arange(buffer_index if buffer_full else BUFFER_SIZE), replace=False, size=BATCH_SIZE)
            batch = {
                "done": buffer[sampled_indices][:, -1],
                "r": buffer[sampled_indices][:, -2],
                "a": buffer[sampled_indices][:, -3].type(torch.int),
                "s": buffer[sampled_indices][:, 0:state_size].reshape([-1] + state_shape),
                "new_s": buffer[sampled_indices][:, state_size:2*state_size].reshape([-1] + state_shape),
            }
            # for index in sampled_indices:
            #     sj, aj, rj, new_sj, donej = buffer[index]
            #     batch["done"].append(float(donej))
            #     batch["r"].append(rj)
            #     batch["s"].append(sj)
            #     batch["new_s"].append(new_sj)
            #     batch["a"].append(aj)

            # batch["done"] = torch.tensor(batch["done"], device=device)
            # batch["r"] = torch.tensor(batch["r"], device=device)
            # batch["s"] = torch.concatenate(batch["s"], dim=0).to(device)
            # batch["new_s"] = torch.concatenate(batch["new_s"], dim=0).to(device)
            # batch["a"] = torch.tensor(batch["a"], dtype=torch.int, device=device)

            with torch.no_grad():
                # target_model for double q learning
                q = model.forward(batch["new_s"].to(device)).max(dim=-1).values
                batch["y"] = batch["r"].to(device) + (1 - batch["done"].to(device)) * GAMMA * q
                batch["y"] = batch["y"].float()

            loss = model.compute_loss(s=batch["s"].to(device), y=batch["y"], a=batch["a"])
            total_loss += loss.item()
            loss.backward()
            model.optimizer.step()
            model.scheduler.step()
            model.optimizer.zero_grad()
            updates_counter += 1
            pbar.update(1)

        if updates_counter == 0:
            continue

        metrics.add_scalar("Epsilon", epsilon, episode_counter)
        metrics.add_scalar('Total Reward per episode', total_episode_reward, episode_counter)
        metrics.add_scalar('Episode length', episode_length, episode_counter)
        metrics.log_system_usage()

        if total_episode_reward > best_total_reward:
            best_total_reward = total_episode_reward
            torch.save(model.state_dict(), "./checkpoints/best_model.pt")

        # if episode_counter % sync_models_frequency == 0:
        #     sync_models(model, target_model)
        
        if episode_counter % 100 == 0:
            torch.save(model.state_dict(), f"./checkpoints/{episode_counter}.pt")

    metrics.close()
    env.close()

if __name__ == "__main__":
    main()
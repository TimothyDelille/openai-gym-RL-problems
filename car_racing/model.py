import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

class DQN(nn.Module):
    def __init__(self, frame_stack_len):
        super().__init__()
        # frame_stack_len is the number of contiguous frames ingested

        # (steering, gas, break)
        # limiting the action space seems to greatly improve training.
        # policy doesn't even learn to turn otherwise.
        # self.action_space = np.array([
        #     [-1, 1, 0], [0, 1, 0], [1, 1, 0],
        #     [-1, 0.5, 0], [0, 0.5, 0], [1, 0.5, 0],
        #     [-1, 0, 0.2], [0, 0, 0.2], [1, 0, 0.2],
        #     [-1, 0, 0], [0, 0, 0], [1, 0, 0]
        # ])
        self.action_space = np.array([
            [-1, 1, 0.2], [0, 1, 0.2], [1, 1, 0.2],
            [-1, 1,   0], [0, 1,   0], [1, 1,   0],
            [-1, 0.5, 0.2], [0, 0.5, 0.2], [1, 0.5, 0.2],
            [-1, 0.5,   0], [0, 0.5,   0], [1, 0.5,   0],
            [-1, 0, 0.2], [0, 0, 0.2], [1, 0, 0.2],
            [-1, 0,   0], [0, 0,   0], [1, 0,   0]
        ])

        self.n_actions = len(self.action_space)
        self.conv1 = nn.Conv2d(frame_stack_len, 6, (7, 7), stride=3)
        self.conv2 = nn.Conv2d(6, 12, (4, 4), stride=1)
        self.lin1 = nn.Linear(300, 128)
        self.lin2 = nn.Linear(128, self.n_actions)

        # image preprocessing steps.
        self.transforms = T.Compose([
            T.ToTensor(),  # scale to [0, 1] adds batch dim
            # crop removes the black footer and crops the image to 84x84
            T.Lambda(lambda img: T.functional.crop(img, top=0, left=6, height=84, width=84)),
            T.Grayscale(),
        ])
    
    def forward(self, x):
        batch_size = x.shape[0]
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = out.reshape(batch_size, -1)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out  # batch_size, n_actions

    def act(self, epsilon, state):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.n_actions-1)

        with torch.no_grad(): 
            q = self.forward(state)
            a = int(q.argmax(-1)[0])  # select first element to eliminate batch dim
            return a
    
    def compute_loss(self, s, y, a):
        out = self.forward(s)
        q = out[np.arange(out.shape[0]), a]
        return F.mse_loss(q, y, reduction="mean")
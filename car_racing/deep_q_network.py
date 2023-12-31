import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

class DQN(nn.Module):
    def __init__(self, frame_stack_len=3, learning_rate=1e-3):
        super().__init__()
        # frame_stack_len is the number of contiguous frames ingested

        # (steering, gas, break)
        # limiting the action space seems to greatly improve training.
        # policy doesn't even learn to turn otherwise.
        self.action_space = np.array([
            [-1, 1,   0], [0, 1,   0], [1, 1,   0],
            [-1, 0.5,   0], [0, 0.5,   0], [1, 0.5,   0],
            [-1, 0, 0.2], [0, 0, 0.2], [1, 0, 0.2],
            [-1, 0,   0], [0, 0,   0], [1, 0,   0]
        ])

        self.n_actions = len(self.action_space)
        self.conv1 = nn.Conv2d(frame_stack_len, 6, (7, 7), stride=3)
        self.conv2 = nn.Conv2d(6, 12, (4, 4), stride=1)
        self.lin1 = nn.Linear(300, 128)
        self.lin2 = nn.Linear(128, 1)
        
        # image preprocessing steps.
        self.preprocess = T.Compose([
            T.ToTensor(),  # scale to [0, 1] adds batch dim
            # crop removes the black footer and crops the image to 84x84
            T.Lambda(lambda img: T.functional.crop(img, top=0, left=6, height=84, width=84)),
            T.Grayscale(),
            # T.Resize((16, 16), antialias=False),
        ])

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9999)
    
    def forward(self, x):
        batch_size = x.shape[0]
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = out.reshape(batch_size, -1)

        q = F.relu(self.lin1(out))
        q = self.lin2(q)
        return q  # batch_size, n_actions

    def act(self, epsilon, state):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.n_actions-1)

        with torch.no_grad(): 
            q = self.forward(state)
            a = int(q.argmax(-1)[0])  # select first element to eliminate batch dim
            return a
    
    def compute_loss(self, s, y, a):
        out = self.forward(s)  # batch, n_actions
        mask = torch.zeros_like(out)
        mask[np.arange(out.shape[0]), a] = 1
        q = (out * mask).sum(-1)
        return F.mse_loss(q, y, reduction="mean")
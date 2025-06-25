import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

from utils import get_conv2d_output_shape

class DQN(nn.Module):
    def __init__(self, frame_stack_len=3, learning_rate=1e-3):
        super().__init__()
        # frame_stack_len is the number of contiguous frames ingested

        # DISCRETE ACTION SPACE
        # each tuple represents (steering, gas, break)
        # limiting the action space seems to greatly improve training.
        # policy doesn't even learn to turn otherwise.
        self.action_space = np.array([
            [-1, 1,   0], [0, 1,   0], [1, 1,   0],
            [-1, 0.5,   0], [0, 0.5,   0], [1, 0.5,   0],
            [-1, 0, 0.2], [0, 0, 0.2], [1, 0, 0.2],
            [-1, 0,   0], [0, 0,   0], [1, 0,   0]
        ])
        self.n_actions = len(self.action_space)
        h_in, w_in = [84, 84]
        c_in = frame_stack_len
        c_out = 16
        kernel = [8, 8]
        stride = [4, 4]
        padding = [0, 0]
        dilation = [1, 1]
        self.conv1 = nn.Conv2d(
            in_channels=c_in, 
            out_channels=c_out, 
            kernel_size=kernel, 
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        h_out, w_out = get_conv2d_output_shape(
            h_in=h_in, 
            w_in=w_in, 
            padding=padding, 
            dilation=dilation,
            kernel=kernel,
            stride=stride
        )
        c_in = c_out
        c_out = 32
        kernel = [4, 4]
        stride = [2, 2]
        self.conv2 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation)
        h_out, w_out = get_conv2d_output_shape(
            h_in=h_out, 
            w_in=w_out,
            padding=padding,
            dilation=dilation,
            kernel=kernel,
            stride=stride,
        )

        hidden_dim = 256
        self.lin1 = nn.Linear(h_out * w_out * c_out, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, self.n_actions)
        
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
        out = F.relu(self.conv2(out))
        out = out.reshape(batch_size, -1)
        out = F.relu(self.lin1(out))
        return F.relu(self.lin2(out))

    def act(self, epsilon, state):
        if random.uniform(0, 1) < epsilon:
            return random.randint(0, self.n_actions-1)

        with torch.no_grad(): 
            q = self.forward(state)
            a = int(q.argmax(-1)[0])  # select first element to eliminate batch dim
            return a
    
    def compute_loss(self, s, y, a):
        """
        a is the action taken.
        we only compute the loss for the action that we ended up taking.
        """
        out = self.forward(s)  # batch, n_actions
        mask = torch.zeros_like(out)
        mask[np.arange(out.shape[0]), a] = 1
        q = (out * mask).sum(-1)
        return F.mse_loss(q, y, reduction="mean")
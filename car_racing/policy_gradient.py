import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

class Policy(nn.Module):
    def __init__(self, frame_stack_len=3, learning_rate=1e-3):
        super().__init__()
        # frame_stack_len is the number of contiguous frames ingested

        self.n_actions = 3  # (steering, gas, break)
        self.conv1 = nn.Conv2d(frame_stack_len, 6, (7, 7), stride=3)
        self.conv2 = nn.Conv2d(6, 12, (4, 4), stride=1)
        self.lin1 = nn.Linear(300, 128)
        self.lin2_mean = nn.Linear(128, self.n_actions) # mean for each action.
        self.lin2_cov = nn.Linear(128, self.n_actions) # std for each action.

        # image preprocessing steps.
        self.preprocess = T.Compose([
            T.ToTensor(),  # scale to [0, 1] adds batch dim
            # crop removes the black footer and crops the image to 84x84
            T.Lambda(lambda img: T.functional.crop(img, top=0, left=6, height=84, width=84)),
            T.Grayscale(),
            # T.Resize((16, 16), antialias=False),
        ])

        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9999)
    
    def forward(self, x):
        batch_size = x.shape[0]
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = out.reshape(batch_size, -1)
        out = F.relu(self.lin1(out))
        mean = self.lin2_mean(out)  # (3,)
        cov = self.lin2_cov(out)  # (3,)
        # apply softplus function to ensure positive std.
        # is also smooth, more stable than exp since it grows slower, and its gradient (the sigmoid function) is well behaved.
        cov = torch.log(1 + torch.exp(cov))
        return mean, cov

    # def act(self, mean, cov): 
    #     eps = np.random.normal(0, 1, size=self.n_actions)
    #     a = mean + torch.sqrt(cov) * torch.tensor(eps, requires_grad=False) # (1, 3)
    #     return a.flatten()
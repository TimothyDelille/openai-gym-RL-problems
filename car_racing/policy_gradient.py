import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils import get_conv2d_output_shape

class Policy(nn.Module):
    def __init__(self, frame_stack_len=3, learning_rate=1e-3):
        super().__init__()
        # frame_stack_len is the number of contiguous frames ingested

        self.n_actions = 3  # (steering, gas, break)

        h_in = w_in = 16
        # image preprocessing steps.
        self.preprocess = T.Compose([
            T.ToTensor(),  # scale to [0, 1] and adds batch dim
            # crop removes the black footer and crops the image to 84x84
            T.Lambda(lambda img: T.functional.crop(img, top=0, left=6, height=84, width=84)),
            T.Grayscale(),
            T.Resize((h_in, w_in), antialias=False),
        ])

        padding = [0, 0]
        dilation = [1, 1]
        kernel = [3, 3]
        stride = [1, 1]
        c_in = frame_stack_len
        c_out = 6
        h_out, w_out = get_conv2d_output_shape(h_in=h_in, w_in=w_in, kernel=kernel, stride=stride, padding=padding, dilation=dilation)
        self.conv_1 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel, stride=stride)
        
        kernel = [2, 2]
        stride = [1, 1]
        h_in, w_in = h_out, w_out
        h_out, w_out = get_conv2d_output_shape(h_in=h_in, w_in=w_in, kernel=kernel, stride=stride, padding=padding, dilation=dilation)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding, dilation=dilation)
        
        c_in = c_out
        c_out = 12
        stride = [1, 1]
        kernel = [2, 2]
        h_in, w_in = h_out, w_out
        h_out, w_out = get_conv2d_output_shape(h_in=h_in, w_in=w_in, kernel=kernel, stride=stride, padding=padding, dilation=dilation)
        self.conv_2 = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=kernel, stride=stride)
        
        kernel = [2, 2]
        stride = [1, 1]
        h_in, w_in = h_out, w_out
        h_out, w_out = get_conv2d_output_shape(h_in=h_in, w_in=w_in, kernel=kernel, stride=stride, padding=padding, dilation=dilation)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding, dilation=dilation)

        self.lin_1 = nn.Linear(h_out * w_out * c_out, 128)
        self.lin_2_mean = nn.Linear(128, self.n_actions) # mean for each action.
        self.lin_2_cov = nn.Linear(128, self.n_actions) # std for each action.
        self.lin_2_value = nn.Linear(128, 1)  # value function.
        
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=learning_rate)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9999)
    
    def forward(self, x):
        batch_size = x.shape[0]
        out = F.relu(self.conv_1(x))
        out = self.max_pool_1(out)
        out = F.relu(self.conv_2(out))
        out = self.max_pool_2(out)
        out = out.reshape(batch_size, -1)
        out = F.relu(self.lin_1(out))
        mean = self.lin_2_mean(out)  # (3,)
        mean = F.sigmoid(mean)
        cov = self.lin_2_cov(out)  # (3,)
        # apply softplus function to ensure positive std.
        # is also smooth, more stable than exp since it grows slower, and its gradient (the sigmoid function) is well behaved.
        cov = torch.log(1 + torch.exp(cov))
        # compute value function. We'll substract it from the returns as a baseline to reduce variance.
        v = self.lin_2_value(out)
        return mean, cov, v

    # def act(self, mean, cov): 
    #     eps = np.random.normal(0, 1, size=self.n_actions)
    #     a = mean + torch.sqrt(cov) * torch.tensor(eps, requires_grad=False) # (1, 3)
    #     return a.flatten()

    
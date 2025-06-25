import math


# helper function to get output shape of Conv2D layer:
# https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
def get_conv2d_output_shape(h_in, w_in, padding, dilation, kernel, stride):
    h_out = math.floor((h_in + 2*padding[0] - dilation[0] * (kernel[0] - 1) - 1)/stride[0] + 1)
    w_out = math.floor((w_in + 2*padding[1] - dilation[1] * (kernel[1] - 1) - 1)/stride[1] + 1)
    return (h_out, w_out)

def average_gradient_size(model):
        total_gradient_norm = 0.0
        num_parameters = 0

        for param in model.parameters():
            if param.grad is not None:
                total_gradient_norm += param.grad.data.norm(2).item()  # 2-norm of the gradient
                num_parameters += 1

        return total_gradient_norm / num_parameters
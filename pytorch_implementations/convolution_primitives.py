import torch
from torch import nn
import math
import torch.nn.functional as F

"""
since i'm writing to a pre allocated tensor each time, these functions are autograd-fragile. see cnn.py for better implementation. 
this is also inefficient. therefore i switched to kernel looping based on offsets, instead of output looping. see cnn.py

"""

def conv1d_manual(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, dilation: int, padding: int) -> torch.Tensor:
    ## handling padding first
    padding_tensor = torch.zeros((padding,))
    x_train = torch.cat([padding_tensor, x, padding_tensor], dim=0)

    ## calculate output length
    K = w.shape[0]
    L_out = math.floor((x.shape[0] + (2 * padding) - (dilation * (K - 1)) - 1) / stride) + 1


    output = torch.zeros((L_out,))
    for i in range(L_out):
        start_index = i * stride
        x_c = x_train[start_index : start_index + (K * dilation) : dilation]
        output[i] = (torch.dot(x_c,w) + b)
    return output
 

def conv2d_manual(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, dilation: int, padding: int) -> torch.Tensor:

    ## handling padding first
    x_train = F.pad(x, (padding, padding, padding, padding))

    C_in, H_in, W_in, K = x.shape[1], x.shape[2], x.shape[3], w.shape[2]
    H_out = math.floor((H_in + (2 * padding) - (dilation * (K - 1)) - 1) / stride) + 1
    W_out = math.floor((W_in + (2 * padding) - (dilation * (K - 1)) - 1) / stride) + 1

    output = torch.zeros((x_train.shape[0], w.shape[0], H_out, W_out))

    for h_i in range(H_out):
        for w_i in range(W_out):
            h_start_index = h_i * stride
            w_start_index = w_i * stride

            x_c = x_train[:,:,
                            h_start_index : h_start_index + (K * dilation) : dilation,
                            w_start_index : w_start_index + (K * dilation) : dilation].unsqueeze(dim=1) ### shape: (N, 1, C_in, K, K)
            output[:,:,h_i,w_i] = (x_c * w.unsqueeze(dim=0)).sum(dim=(2,3,4)) + b ### shape: (N, 1, C_in, K, K) * (1, C_out, C_in, K, K)
    
    return output     
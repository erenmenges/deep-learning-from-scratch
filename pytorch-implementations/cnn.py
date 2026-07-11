import torch
from torch import nn
import torch.nn.functional as F
import math



def conv2d_forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, stride: int, padding: int) -> torch.Tensor:

    ## handling padding first
    x_train = F.pad(x, (padding, padding, padding, padding))
    
    N, C_out, H_in, W_in, K, C_in = x.shape[0], w.shape[0], x.shape[2], x.shape[3], w.shape[2], x.shape[1]

    H_out = math.floor((H_in + (2 * padding) - K) / stride) + 1
    W_out = math.floor((W_in + (2 * padding) - K) / stride) + 1
    output = torch.zeros((N, C_out, H_out, W_out))


    for m in range(K):
        for n in range(K):
            x_c = x_train[:,:, m : m + (H_out * stride) : stride, n : n + (W_out * stride) : stride]
            output += (w[:,:, m, n][None, :,:, None, None] * x_c[:, None, ...]).sum(dim=2) 
    output = output + b[None, :, None, None]
    
    return output

class conv2d(nn.Module):
    def __init__(self, C_out: int, C_in: int, K: int, stride: int, padding: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(C_out, C_in, K, K) * 0.1)
        self.b = nn.Parameter(torch.zeros((C_out)))
        self.stride = stride
        self.padding = padding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv2d_forward(x, self.w, self.b, self.stride, self.padding)
    

class whole_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(conv2d(8,1,3,1,1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     conv2d(16,8,3,1,1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Flatten(),
                                     nn.Linear(in_features=64, out_features=10) )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

# data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

d = load_digits()
X = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16.0  ### shape: (1797, 1, 8, 8)
y = torch.tensor(d.target, dtype=torch.long)                          ### shape: (1797,)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)



model = whole_model()

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

epochs = 1000

for epoch in range(epochs):
    model.train()

    y_hat = model(Xtr)

    loss = loss_fn(y_hat, ytr)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    with torch.inference_mode():
        if epoch % 125 == 0:
            model.eval()
            test_forward = model(Xte)
            test_loss = loss_fn(test_forward, yte)
            print(f"Epoch: {epoch}, Train loss: {loss.item()}, Test loss: {test_loss.item()}")


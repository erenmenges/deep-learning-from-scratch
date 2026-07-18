import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import random

class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            subset_mask = torch.bernoulli(torch.full_like(x, (1 - self.p)))
            return x * subset_mask
        else:
            return x * (1 - self.p)

class MLP(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features=64,out_features=64), nn.ReLU(), Dropout(p), 
                                    nn.Linear(in_features=64,out_features=64), nn.ReLU(), Dropout(p), nn.Linear(in_features=64, out_features=10))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
def make_batches(n: int, batch_size: int, seed: int):
    batches = []
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)
    for i in range(0,n,batch_size):
        batches.append(indices[i:i+batch_size])
    return batches




def train(model: nn.Module) -> tuple[float,float]:


    X, Y = load_digits(return_X_y=True)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=42)
    X_train, X_test, Y_train, Y_test = torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(Y_train).long(), torch.from_numpy(Y_test).long()

    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    epoch_mean_losses = []

    for epoch in range(100):
        batches = make_batches(n=X_train.shape[0], batch_size=20, seed=42)
        
        ## train mode
        model.train()
        epoch_batch_losses = []
    
        for batch in batches:
            ## batches
            X_train_batch = X_train[batch]
            Y_train_batch = Y_train[batch]


            ## forward pass
            Y_pred = model(X_train_batch)

            ## calculate the loss
            loss = loss_fn(Y_pred, Y_train_batch)
            epoch_batch_losses.append(loss.item())

            ## optim zero grad
            optimizer.zero_grad()

            ## backprop
            loss.backward()

            ## step 
            optimizer.step()
        epoch_mean_losses.append(sum(epoch_batch_losses)/len(epoch_batch_losses))

        ## model eval time
        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                test_pred = model.forward(X_test)
                test_loss = loss_fn(test_pred, Y_test)
                print(f"epoch: {epoch}, train loss: {loss}, test loss: {test_loss}")

    return (epoch_mean_losses[0], epoch_mean_losses[-1])

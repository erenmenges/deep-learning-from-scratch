import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from batchnorm import BatchNorm
import time

class ResidualBlock(nn.Module):
    def __init__(self, C: int, mode: str = "standard"):
        super().__init__()
        if mode == "standard":
            self.network = nn.Sequential(BatchNorm(C), nn.ReLU(),
                                          nn.Conv2d(in_channels=C, out_channels=C,
                                                                            kernel_size=3, padding=1, bias=False),
                                        BatchNorm(C), nn.ReLU(),
                                          nn.Conv2d(in_channels=C, out_channels=C,
                                                                            kernel_size=3, padding=1, bias=False))
        # bottleneck block
        else:
            self.network = nn.Sequential(BatchNorm(C), nn.ReLU(),
                                          nn.Conv2d(in_channels=C, out_channels=C//4,
                                                                            kernel_size=1, padding=0, bias=False),
                                        BatchNorm(C//4), nn.ReLU(),
                                          nn.Conv2d(in_channels=C//4, out_channels=C//4,
                                                                            kernel_size=3, padding=1, bias=False),
                                        BatchNorm(C//4), nn.ReLU(),
                                          nn.Conv2d(in_channels=C//4, out_channels=C,
                                                                            kernel_size=1, padding=0, bias=False))
            

    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # add the input to block output
        return h + self.network(h)
    
class PlainBlock(nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.network = nn.Sequential(BatchNorm(C), nn.ReLU(),
                                          nn.Conv2d(in_channels=C, out_channels=C,
                                                                            kernel_size=3, padding=1, bias=False),
                                        BatchNorm(C), nn.ReLU(),
                                          nn.Conv2d(in_channels=C, out_channels=C,
                                                                            kernel_size=3, padding=1, bias=False))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(h)
    


def get_digits_data(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X, y = load_digits(return_X_y=True)

    # train test split
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    # shape them into proper (N, C_in, H, W)
    X_train, X_test, y_train, y_test = torch.from_numpy(X_train).float().reshape((X_train.shape[0], 1, 8, 8)), torch.from_numpy(X_test).float().reshape((X_test.shape[0], 1, 8, 8)), torch.from_numpy(y_train).long(), torch.from_numpy(y_test).long()

    # standardize
    mu = X_train.mean()
    sigma = X_train.std()
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    return (X_train, y_train, X_test, y_test)


def build_resnet(blocks_per_stage: int = 2, block: str = "standard") -> nn.Module:
    blocks_list = [nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)]  ### stem
    blocks_list.extend([ResidualBlock(C=16, mode = block) for _ in range(blocks_per_stage)])  ### first resblock
    blocks_list.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2))  ### downsample
    blocks_list.extend([ResidualBlock(C=32, mode = block) for _ in range(blocks_per_stage)])  ### second resblock
    blocks_list.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2))  ### downsample again
    blocks_list.extend([ResidualBlock(C=64, mode = block) for _ in range(blocks_per_stage)])  ### third resblock
    blocks_list.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_features=64, out_features=10)])  ### avgpool and fully connected layer

    # * unpacks the list
    return nn.Sequential(*blocks_list)

def build_plain(blocks_per_stage: int = 2) -> nn.Module:
    blocks_list = [nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)]  ### stem
    blocks_list.extend([PlainBlock(C=16) for _ in range(blocks_per_stage)])  ### first plain block
    blocks_list.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2))  ### downsample
    blocks_list.extend([PlainBlock(C=32) for _ in range(blocks_per_stage)])  ### second plain block
    blocks_list.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2))  ### downsample again
    blocks_list.extend([PlainBlock(C=64) for _ in range(blocks_per_stage)])  ### third plain block
    blocks_list.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_features=64, out_features=10)])  ### adaptive avgpool and fully connected layer

    # * unpacks the list
    return nn.Sequential(*blocks_list)

def make_batches(X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int = 64) -> list:
    # shuffle the training data
    permutated_indices = torch.randperm(X_train.shape[0])
    X_permutated = X_train[permutated_indices]
    y_permutated = y_train[permutated_indices]
    batches = []

    # make batches
    for i in range(0, X_train.shape[0], batch_size):
        batches.append((X_permutated[i : i + batch_size], y_permutated[i : i + batch_size]))
    return batches


def freeze_batchnorms_for_eval(model: nn.Module, X_train: torch.Tensor) -> None:
    for module in model.modules():
        if isinstance(module, BatchNorm):
            module.capture = True

    model.train()
    with torch.no_grad():
        model(X_train)
    
    for module in model.modules():
        if isinstance(module, BatchNorm):
            module.capture = False
    

def train_model(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor,
           y_test: torch.Tensor, epochs: int = 30, batch_size: int = 64, lr: float = 0.05, seed: int = 0) -> dict:
    
    # tensors are small, so coordination between threads costs more. changing this made training go from 25s to 14s bps=2
    torch.set_num_threads(1)

    t0 = time.perf_counter()
    torch.manual_seed(seed)

    # sgd with momentum, cross entropy loss (softmax inside as well)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    # prepare stem grad variable
    g = None

    for epoch in range(epochs):
        batches = make_batches(X_train, y_train, batch_size)
        model.train()
        batch_losses = None
        ## prepare for last epochs mean loss
        if epoch == epochs - 1:
            batch_losses = []

        ## actually train
        for b,batch in enumerate(batches):
            y_hat = model(batch[0])
            loss = loss_fn(y_hat, batch[1])
            if batch_losses is not None:
                batch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            ## save stem gradient first epoch first batch
            if epoch == 0 and b == 0:
                g = model[0].weight.grad
                g = g.norm().item()

            ## backprop
            optimizer.step()
    
        ## eval every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                ## freeze statistics for final eval
                freeze_batchnorms_for_eval(model, X_train)
                
                ## now eval
                model.eval()
                logits = model(X_test)
                test_preds = logits.argmax(dim=1)
                test_accuracy = (test_preds == y_test).float().mean().item()
                test_loss = loss_fn(logits, y_test)

                print(f"Epoch: {epoch}, Train loss: {loss}, Test loss: {test_loss}, Test accuracy: {test_accuracy}")
    

    # freeze statistics for final eval
    freeze_batchnorms_for_eval(model, X_train)

    # final test accuracy measurement
    with torch.no_grad():
        model.eval()
        logits = model(X_test)
        test_preds = logits.argmax(dim=1)
        test_accuracy = (test_preds == y_test).float().mean().item()
    
    print(f"Time spent training: {time.perf_counter() - t0}s")
    return {"final_train_loss": sum(batch_losses) / len(batch_losses), "test_acc": test_accuracy, "stem_grad_norm": g}

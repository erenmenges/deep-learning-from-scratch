import torch
from torch import nn
from mini_GPT import TransformerBlock

torch.manual_seed(0)
torch.set_num_threads(1)

def patchify(images: torch.Tensor) -> torch.Tensor:
    intermediate = images.reshape(images.shape[0], 4, 2, 4, 2)  ### (B, row of patch, row inside patch, col of patch, col inside patch)
    intermediate = intermediate.permute(0, 1, 3, 2, 4)
    intermediate = intermediate.reshape(images.shape[0], 16, 4)
    return intermediate


class VisionTransformer(nn.Module):
    def __init__(self, K: int, D: int, D_q: int, H: int, mask: bool):
        super().__init__()
        self.patch_embeddings = nn.Linear(in_features=4, out_features=D)
        self.positional_embeddings = nn.Parameter(torch.randn(17, D))
        self.cls = nn.Parameter(torch.randn(D))

        layers = []
        for _ in range(K):
            layers.append(TransformerBlock(D, D_q, H, mask))
        
        self.main_layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(in_features=D, out_features=10)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoded_X = self.patch_embeddings(X)  ### (B, 16, 4) --> (B, 16, D)
        prepended_X = torch.cat((self.cls.expand(encoded_X.shape[0], 1, self.cls.shape[0]), encoded_X), dim=1)  ### (B, 16, D) --> (B, 17, D)
        positionally_embedded_X = prepended_X + self.positional_embeddings[None, :, :]  ### (B, 17, D) = (B, 17, D) + (1, 17, D)
        positionally_embedded_X = positionally_embedded_X.permute(0, 2, 1)  ### (B, D, 17)
        main = self.main_layers(positionally_embedded_X)  ### (B, D, 17)
        main = main.permute(0, 2, 1)  ### (B, 17, D)
        output = self.output_head(main[:, 0, :])  ### (B, 1, D) --> (B, D) ---> (B, 10)
        return output  ### (B, 10)
    
def make_batches(N: int, batch_size: int) -> list:
    permutated_indices = torch.randperm(N)
    batches = []
    for i in range(0, N, batch_size):
        batches.append(permutated_indices[i : i + batch_size])
    return batches


def train_vit(X_train, y_train) -> VisionTransformer:
    epochs = 150
    model = VisionTransformer(2, 32, 8, 4, False)
    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    patchified_X = patchify(X_train)

    for epoch in range(epochs):
        batches = make_batches(patchified_X.shape[0], 32)
        for b in batches:
            model.train()

            X_batch = patchified_X[b]
            y_batch = y_train[b]

            y_hat = model(X_batch)

            loss = loss_fn(y_hat, y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, Training loss: {loss}")

    return model
        

def vit_predict(model: VisionTransformer, X: torch.Tensor) -> torch.LongTensor:
    model.eval()
    with torch.no_grad():
        return torch.argmax(model(patchify(X)), dim=1)
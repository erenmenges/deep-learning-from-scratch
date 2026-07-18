import torch
import math
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(1)

class TransformerBlock(nn.Module):
    def __init__(self, D: int, D_q: int, H: int, mask: bool):
        super().__init__()

        self.heads = H
        self.mask = mask

        self.W_keys = nn.Parameter(torch.randn(H, D_q, D))
        self.b_keys = nn.Parameter(torch.randn(H, D_q, 1))

        self.W_queries = nn.Parameter(torch.randn(H, D_q, D))
        self.b_queries = nn.Parameter(torch.randn(H, D_q, 1))

        self.W_values = nn.Parameter(torch.randn(H, D_q, D))
        self.b_values = nn.Parameter(torch.randn(H, D_q, 1))

        self.W_linear_at_end = nn.Parameter(torch.randn(D, D))

        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(in_features=D, out_features=4*D), nn.ReLU(), nn.Linear(in_features=4*D, out_features=D))

    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        params = {"Omega_v": self.W_values, "Omega_q": self.W_queries, "Omega_k": self.W_keys, "Omega_c": self.W_linear_at_end, 
                  "omega_v": self.b_values, "omega_q": self.b_queries, "omega_k": self.b_keys}
        
        step1 = X + multihead_self_attention(X, params, self.heads, self.mask)


        step2 = step1.permute(0, 2, 1)  ### make it from (B, D, N) to (B, N, D)
        step2 = self.ln1(step2)

        step3 = self.mlp(step2)
        step3 = step3.permute(0, 2, 1) + X ### return it back to (B, D, N) and add residual

        step4 = step3.permute(0, 2, 1) 
        step4 = self.ln2(step4)
        step4 = step4.permute(0, 2, 1)
        return step4


class Transformer(nn.Module):
    def __init__(self, K: int, D: int, D_q: int, H: int, V: int, N: int, mask: bool):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(D, V))
        self.positional_encoding = nn.Parameter(torch.randn(D, N))

        layers = []
        for _ in range(K):
            layers.append(TransformerBlock(D, D_q, H, mask))
        
        self.main_layers = nn.Sequential(*layers)
        self.output_head = nn.Linear(in_features=D, out_features=V)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoded_X = self.embeddings.T[X]
        encoded_X = encoded_X.transpose(1,2)
        main = self.main_layers(encoded_X + self.positional_encoding[:, :encoded_X.shape[-1]])
        main = main.permute(0, 2, 1)
        output = self.output_head(main)
        return output.permute(0, 2, 1)


def multihead_self_attention(X: torch.Tensor, params: dict, H: int, mask: bool) -> torch.Tensor:
    W_v, W_q, W_k, W_c  = params["Omega_v"], params["Omega_q"], params["Omega_k"], params["Omega_c"]
    b_v, b_q, b_k = params["omega_v"], params["omega_q"], params["omega_k"]

    N = X.shape[-1]
    D = W_q.shape[2]
    D_h = W_q.shape[1]
    B = X.shape[0]


    V = (W_v[None, :, :, :] @ X[:, None, :, :]) + b_v[None, :, :, :]  ### shape: (B, H, D_h, N) = ((1, H, D_h, D) @ (B, 1, D, N)) + (1, H, D_h, 1)
    Q = (W_q[None, :, :, :] @ X[:, None, :, :]) + b_q[None, :, :, :]  ### shape: (B, H, D_h, N) = ((1, H, D_h, D) @ (B, 1, D, N)) + (1, H, D_h, 1)
    K = (W_k[None, :, :, :] @ X[:, None, :, :]) + b_k[None, :, :, :]

    raw_attention = (K.mT @ Q)  ### shape: (B, H, N, N) = (B, H, N, D_h) @ (B, H, D_h, N)
    if mask:
        bool_mask = torch.tril(torch.ones((B, H, N, N), device=raw_attention.device), diagonal=-1).bool()
        raw_attention.masked_fill_(bool_mask, float('-inf'))

    head_outputs = V @ F.softmax((raw_attention/math.sqrt(W_q.shape[1])), dim=-2)  ### shape: (B, H, D_h, N) = (B, H, D_h, N) @ (B, H, N, N)
    outputs_stacked = head_outputs.reshape(B, -1, head_outputs.shape[-1])  ### shape: (B, D, N) = (B, H * D_h, N)

    final_output = W_c[None, :, :] @ outputs_stacked  ### shape: (B, D, N) = (1, D, D) @ (B, D, N)

    return final_output


def make_batch(B, seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, 10, (B, 3), generator=g)
    return base.repeat(1, 4)          # (B, 12): t1 t2 t3 t1 t2 t3 ...


def train_lm():
    model = Transformer(2, 32, 8, 4, 11, 13, True)
    epochs = 1000

    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        batch = make_batch(20, epoch)
        batch = torch.cat([torch.full((batch.shape[0], 1), 10), batch], dim=1)

        model.train()

        logits = model(batch)[:,:,:-1]  ### shape: (B, V, N-1), last dim is everything except last token
        targets = batch[:,1:] ### shape: (B, N-1), last dim is everything except first token

        logits_for_ce = logits.mT.reshape(-1, logits.shape[1])  ### mT makes it (B, N-1, V), reshape makes it (B * (N-1), V)
        targets_for_ce = targets.reshape(-1)  ### makes it shape (B * N-1)


        loss = loss_fn(logits_for_ce, targets_for_ce)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                model.eval()

                test_batch = make_batch(20, epoch * -1)
                test_batch = torch.cat([torch.full((test_batch.shape[0], 1), 10), test_batch], dim=1)

                test_logits = model(test_batch)[:,:,:-1]
                test_targets = test_batch[:,1:]

                test_logits_for_ce = test_logits.mT.reshape(-1, test_logits.shape[1])  ### mT makes it (B, N-1, V), reshape makes it (B * (N-1), V)
                test_targets_for_ce = test_targets.reshape(-1)  ### makes it shape (B * N-1)

                test_loss = loss_fn(test_logits_for_ce, test_targets_for_ce)

                print(f"Epoch: {epoch}, Training loss: {loss}, Test loss: {test_loss}")
    return model


def lm_logits(model: Transformer, tokens: torch.Tensor) -> torch.Tensor:
    return model(tokens).permute(0, 2, 1)


def greedy_generate(model: Transformer, prompt: torch.Tensor, n_steps: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        context = torch.clone(prompt)
        for _ in range(n_steps):
            context = torch.column_stack((context, torch.argmax(model(context)[:, :, -1], dim=1)))
        return context

#model = train_lm()
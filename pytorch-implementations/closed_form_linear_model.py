import torch
import math
import matplotlib.pyplot as plt

def hidden_features(x: torch.Tensor, D: int) -> torch.Tensor:
    bias_column = torch.ones(x.shape[0]).unsqueeze(dim=1)
    weights_matrix = torch.zeros((x.shape[0], D))
    indices = torch.arange(weights_matrix.shape[1], dtype=torch.float32).view(1,-1)
    weight = (x.unsqueeze(dim=1) - indices/D)
    weights_matrix = torch.maximum(weights_matrix, weight)
    return torch.cat((bias_column, weights_matrix), dim=1)

def model_forward(x: torch.Tensor, omega: torch.Tensor, D: int) -> torch.Tensor:
    return hidden_features(x, D) @ omega

def fit_closed_form(x: torch.Tensor, y: torch.Tensor, D: int) -> torch.Tensor:
    phi = hidden_features(x, D)
    return torch.linalg.lstsq(phi, y.unsqueeze(1)).solution.squeeze(1)


def true_mean(x):            
    return torch.sin(2*math.pi*x)   ## arbitrary

def sample_dataset(N: int, sigma: float) -> tuple[torch.Tensor, torch.Tensor]:
    samples = (torch.arange(N) + torch.rand(N)) / N
    noise = torch.randn(samples.shape)
    return (samples, true_mean(samples) + (sigma * noise))

def bias_variance_noise(x_eval: torch.Tensor, D: int, N: int, sigma: float, n_datasets: int, n_y: int) -> dict:
    weights = []
    for _ in range(n_datasets):
        samples = sample_dataset(N, sigma)
        weights.append(fit_closed_form(samples[0], samples[1], D).unsqueeze(dim=1))
    weights = torch.column_stack(weights)
    predictions = model_forward(x_eval, weights, D).T
    means = torch.mean(predictions, dim=0)

    bias_sq = torch.square((means - true_mean(x_eval))).mean().item()
    variance = ((predictions - means)**2).mean(dim=0).mean().item()

    y_eval = []
    for _ in range(n_y):
        noise = torch.randn_like(x_eval)
        y_eval.append(true_mean(x_eval) + sigma * noise)
    y_eval = torch.column_stack(y_eval).T

    
    individual_losses = (predictions.unsqueeze(1).broadcast_to(predictions.shape[0], n_y, predictions.shape[1]) - y_eval.unsqueeze(0)) ** 2
    model_avg_loss = individual_losses.mean(dim=(0,1)).mean().item()

    sigma_sq = sigma ** 2
    return {"bias2":bias_sq,"variance":variance, "noise":sigma_sq,"expected_loss":model_avg_loss}



def tradeoff_curve(capacities: list[int], x_eval: torch.Tensor, N:int, sigma:float, n_datasets: int) -> dict:
    total_tensor = torch.zeros((len(capacities),))
    variance_tensor = torch.zeros((len(capacities),))
    bias_sq_tensor = torch.zeros((len(capacities),))


    for i, capacity in enumerate(capacities):
        dct = bias_variance_noise(x_eval, capacity, N, sigma, n_datasets, 10)
        variance_tensor[i] = dct["variance"]
        bias_sq_tensor[i] = dct["bias2"]
        total_tensor[i] = dct["bias2"] + dct["variance"]
    
    fig, ax = plt.subplots()

    total_tensor = total_tensor.tolist() 
    variance_tensor = variance_tensor.tolist() 
    bias_sq_tensor = bias_sq_tensor.tolist() 

    ax.plot(capacities, total_tensor)
    ax.plot(capacities, variance_tensor)
    ax.plot(capacities, bias_sq_tensor)
    
    plt.show()
    return {"bias2":bias_sq_tensor, "variance":variance_tensor, "total": total_tensor}
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def alpha_t(t):
    return t

def sigma_t(t):
    return 1 - t

def dalpha_dt(t):
    return torch.ones_like(t)

def dsigma_dt(t):
    return -torch.ones_like(t)

means = {
    1: np.array([-2, 0]),
    2: np.array([2, 0]),
    3: np.array([0, 2 * np.sqrt(3)])
}
cov = np.eye(2)

def sample_mixture(batch_size=1):
    y = np.random.choice([1, 2, 3], size=batch_size)
    x = np.array([np.random.multivariate_normal(mean=means[label], cov=cov) for label in y])

    return x, y

def sample_noise(batch_size=1):
    return np.random.normal(size=(batch_size, 2))

def sample_time(batch_size=1):
    return np.random.uniform(0, 1, size=(batch_size,))

batch_size = 256
num_steps = 5000
lr = 1e-3

network = MLP(x_dim=2, y_num_classes=4, y_emb_dim=16, hidden_dim=256).to(device)
optimizer = optim.Adam(network.parameters(), lr=lr)
loss_fn = nn.MSELoss()

os.makedirs("models", exist_ok=True)

network.train()

def training_guided_flows(p_unconditional):
    for step in range(num_steps):
        # 1. Sample data
        x1, y = sample_mixture(batch_size = batch_size)
        x1_batch = torch.tensor(x1, dtype=torch.float32).to(device)

        # 2. Conditional or unconditional
        y_batch = torch.tensor(y, dtype=torch.long).to(device) # shape (batch_size,)
        mask = torch.rand(batch_size) < p_unconditional  # True tamo gde želimo unconditional
        y_batch[mask] = 0  

        # 3. Sample noise and time
        x0 = sample_noise(batch_size=batch_size)
        x0_batch = torch.tensor(x0, dtype=torch.float32).to(device)
        t = sample_time(batch_size=batch_size)
        t_batch = torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)

        # 4. Calculate x_t and dx/dt
        x_t = alpha_t(t_batch) * x1_batch + sigma_t(t_batch) * x0_batch      # shape (batch_size,2)
        dx_dt = dalpha_dt(t_batch) * x1_batch + dsigma_dt(t_batch) * x0_batch  # shape (batch_size,2)

        # 5. Forward pass
        u_t_pred = network(x_t, t_batch, y_batch)  # shape (batch_size, 2)

        # 6. Loss and optimization step
        loss = loss_fn(u_t_pred, dx_dt)  # MSE loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 1000 == 0:
            print(f"Step {step+1}/{num_steps}, Loss: {loss.item():.6f}")

    torch.save(network.state_dict(), "models/model_final.pt")

if __name__ == "__main__":
    training_guided_flows(p_unconditional=0.1)
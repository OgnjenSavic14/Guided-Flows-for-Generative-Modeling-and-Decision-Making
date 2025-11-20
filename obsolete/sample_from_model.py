from model import MLP
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_model import sample_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu", flush = True)

model = MLP(x_dim=2, y_num_classes=4, y_emb_dim=16, hidden_dim=256).to(device)
state = torch.load("models/model_final_cpu.pt", map_location=device)
model.load_state_dict(state)
model.eval()

def ode_step_rk4(model, x_t, t, h, y, w, device):
    with torch.no_grad():
        y0 = torch.zeros_like(y)
        k1 = (1 - w) * model(x_t, t, y0) + w * model(x_t, t, y)
        k2 = (1 - w) * model(x_t + 0.5 * h * k1, t + 0.5 * h, y0) + w * model(x_t + 0.5 * h * k1, t + 0.5 * h, y)
        k3 = (1 - w) * model(x_t + 0.5 * h * k2, t + 0.5 * h, y0) + w * model(x_t + 0.5 * h * k2, t + 0.5 * h, y)
        k4 = (1 - w) * model(x_t + h * k3, t + h, y0) + w * model(x_t + h * k3, t + h, y)
        x_next = x_t + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next

def sampling_from_guided_flows(model, device, y=0, w=1.0, num_steps=100, batch_size=512):
    model.eval()
    with torch.no_grad():
        x0 = torch.tensor(sample_noise(batch_size=batch_size), dtype=torch.float32, device=device)
        h = 1.0 / num_steps
        t = torch.zeros((batch_size, 1), device=device)
        x_t = x0.clone()
        y_tensor = torch.full((batch_size,), y, dtype=torch.long, device=device)
        for _ in range(num_steps):
            x_t = ode_step_rk4(model, x_t, t, h, y_tensor, w, device)
            t = t + h

    return x_t.detach().cpu().numpy()

n_iters = 2000
batch_size = 512

print("SAMPLING...", flush=True)

# data1 = []
# for i in range(3):
#     y = i + 1
#     cluster_batches = []

#     for j in range(n_iters):
#         samples = sampling_from_guided_flows(
#             model, device,
#             y=y, w=1.0, num_steps=100,
#             batch_size=batch_size
#         )
#         cluster_batches.append(samples)
#         if (j + 1) % 100 == 0:
#             print(f"{i+1}-th cluster, {j+1}-th iteration finished!", flush = True)

#     cluster = np.concatenate(cluster_batches, axis=0)
#     data1.append(cluster)

# print("DATA1", flush=True)

data2 = []
for i in range(3):
    y = i + 1
    cluster_batches = []

    for j in range(n_iters):
        samples = sampling_from_guided_flows(
            model, device,
            y=y, w=4.0, num_steps=100,
            batch_size=batch_size
        )
        cluster_batches.append(samples)
        if (j + 1) % 100 == 0:
            print(f"{i+1}-th cluster, {j+1}-th iteration finished!", flush = True)

    cluster = np.concatenate(cluster_batches, axis=0)
    data2.append(cluster)

print("DATA2", flush=True)

print("PLOTING", flush = True)
fig, ax = plt.subplots(figsize=(8,6))

points = np.vstack(data2)
plt.hist2d(points[:, 0], points[:, 1], bins=500, cmap='binary')

plt.title("u_t")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("plots/u_t_data2.png", dpi=300)
plt.close()
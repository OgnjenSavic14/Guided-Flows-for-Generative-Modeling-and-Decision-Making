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
        y0 = torch.tensor([0], dtype=torch.long, device = device)
        k1 = (1 - w) * model(x_t, t, y0) + w * model(x_t, t, y)
        k2 = (1 - w) * model(x_t + 0.5 * h * k1, t + 0.5 * h, y0) + w * model(x_t + 0.5 * h * k1, t + 0.5 * h, y)
        k3 = (1 - w) * model(x_t + 0.5 * h * k2, t + 0.5 * h, y0) + w * model(x_t + 0.5 * h * k2, t + 0.5 * h, y)
        k4 = (1 - w) * model(x_t + h * k3, t + h, y0) + w * model(x_t + h * k3, t + h, y)
        x_next = x_t + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next

def sampling_from_guided_flows(model, device, y=0, w=1.0, num_steps=100):
    model.eval()
    with torch.no_grad():
        x0 = torch.tensor(sample_noise(batch_size=1), dtype=torch.float32, device=device)
        h = 1.0 / num_steps
        t = torch.zeros_like(x0[:, :1], device=device)
        x_t = x0.clone()

        for _ in range(num_steps):
            x_t = ode_step_rk4(model, x_t, t, h, torch.tensor([y], dtype=torch.long, device=device), w, device)
            t = t + h

    return x_t.squeeze(0).detach().cpu().numpy()


n_samples = 400

print("SAMPLING...", flush = True)

data1 = []
for i in range(3):
    y = i + 1
    cluster = np.array([sampling_from_guided_flows(model, device, y=y, w=1.0, num_steps=100) for _ in range(n_samples)])
    data1.append(np.stack(cluster))

print("DATA1", flush = True)

# data2 = []
# for i in range(3):
#     y = i + 1
#     cluster = np.array([sampling_from_guided_flows(model, device, y=y, w=4.0, num_steps=100) for _ in range(n_samples)])
#     data2.append(np.stack(cluster))

# print("DATA2")

# data3 = []
# for i in range(3):
#     y = i + 1
#     cluster = np.array([sampling_from_guided_flows(model, device, y=y, w=0.0, num_steps=100) for _ in range(n_samples)])
#     data3.append(np.stack(cluster))

# print("DATA3")

print("PLOTING", flush = True)
fig, ax = plt.subplots(figsize=(8,6))

points = np.vstack(data1)
# for i, cluster in enumerate(data1):
#     # ax.scatter(
#     #     cluster[:,0], cluster[:,1],
#     #     s=5,                      # veličina tačke
#     #     alpha=0.4,                # providnost (0.0 - 1.0)
#     #     color=colors[i]
#     # )
#     ax.hist2d(
#         cluster[:, 0], cluster[:, 1],
#         bins=100,                    # broj binova
#         cmap=colors[i],             # svakom klasteru druga mapa boja
#         alpha=0.6,                  # prozirnost da se preklapaju
#         density=True                # da prikazuje gustinu
#     )

plt.hist2d(points[:, 0], points[:, 1], bins=100, cmap='binary')

plt.title("u_t")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("plots/u_t_data1.png", dpi=300)
plt.close()

# fig, ax = plt.subplots(figsize=(8,6))
# colors = ['Red', 'Green', 'Blue']

# for i, cluster in enumerate(data2):
#     ax.scatter(
#         cluster[:,0], cluster[:,1],
#         s=5,                      # veličina tačke
#         alpha=0.4,                # providnost (0.0 - 1.0)
#         color=colors[i]
#     )

# plt.title("u_t")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig("plots/u_t_data2.png", dpi=300)
# plt.close()

# fig, ax = plt.subplots(figsize=(8,6))
# colors = ['Red', 'Green', 'Blue']

# for i, cluster in enumerate(data3):
#     ax.scatter(
#         cluster[:,0], cluster[:,1],
#         s=5,                      # veličina tačke
#         alpha=0.4,                # providnost (0.0 - 1.0)
#         color=colors[i]
#     )

# plt.title("u_t")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig("plots/u_t_data3.png", dpi=300)
# plt.close()
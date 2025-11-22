import numpy as np
import glob
import matplotlib.pyplot as plt

all_samples = sorted(glob.glob("samples/*.npz"))

all_points = []

for sample in all_samples:
    data = np.load(sample)
    data = [data[key] for key in data.files]
    
    all_points.append(np.vstack(data))

fig, axes = plt.subplots(1, len(all_points), figsize=(5 * len(all_points), 5))

for i, points in enumerate(all_points):
    ax = axes[i]
    ax.hist2d(points[:, 0], points[:, 1], bins=500, cmap='binary')

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(f"$\\omega = {i:.1f}$")

plt.tight_layout()
plt.savefig("plots/Figure_3_1.png", dpi=300)
plt.close()

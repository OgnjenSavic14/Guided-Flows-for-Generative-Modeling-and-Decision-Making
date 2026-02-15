import numpy as np

def sample_mixture(batch_size=1):
    means = {
    1: np.array([-2, 0]),
    2: np.array([2, 0]),
    3: np.array([0, 2 * np.sqrt(3)])
    }
    cov = np.eye(2)
    
    y = np.random.choice([1, 2, 3], size=batch_size)
    x = np.array([np.random.multivariate_normal(means[label], cov) for label in y])
    return x, y

def sample_noise(batch_size=128, dim=2):
    return np.random.normal(size=(batch_size, dim))

def sample_time(batch_size=128):
    return np.random.uniform(0, 1, size=(batch_size,))

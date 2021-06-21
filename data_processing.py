import torch
from sklearn.datasets import make_gaussian_quantiles
# Construct dataset

def generate_experiences(n_timesteps, n_channels, n_tasks, n_samples=100, test=False):

    if test:
        n_samples //= 5
    experiences = [make_gaussian_quantiles(mean=[i for _ in range(n_channels*n_timesteps)],
                                    cov=[3. for _ in range(n_channels*n_timesteps)],
                                    n_samples=100, n_features=n_channels*n_timesteps,
                                    n_classes=2, random_state=1)
                                    for i in range(n_tasks)]

    for i in range(n_tasks):
        x = torch.FloatTensor(experiences[i][0]).view(-1, n_timesteps, n_channels)
        y = torch.LongTensor(experiences[i][1])
        experiences[i] = (x,y)

    return experiences

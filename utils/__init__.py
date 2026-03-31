import torch
import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


def get_dataset(name: str):
    """Download and return (train_dataset, test_dataset)."""
    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST("./data", train=True,  download=True, transform=tf)
        test  = datasets.MNIST("./data", train=False, download=True, transform=tf)
    elif name == "cifar10":
        tf_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
        ])
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
        ])
        train = datasets.CIFAR10("./data", train=True,  download=True, transform=tf_train)
        test  = datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return train, test


def partition_data(dataset, n_clients: int, n_samples_per_client: int, seed: int = 42):
    """
    Assign `n_samples_per_client` samples (with replacement allowed across clients)
    to each of `n_clients` clients. Returns list of index arrays.
    Paper: 1000 clients, 200 samples each.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    client_indices = []
    for _ in range(n_clients):
        idx = rng.integers(0, n, size=n_samples_per_client)
        client_indices.append(idx.tolist())
    return client_indices


class PoisonedSubset(Dataset):
    """Wraps a dataset and applies label poisoning to selected indices."""
    def __init__(self, base_dataset, indices, poison_fn=None):
        self.base    = base_dataset
        self.indices = indices
        self.poison_fn = poison_fn  # fn(label) -> new_label, or None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base[self.indices[i]]
        if self.poison_fn is not None:
            y = self.poison_fn(y)
        return x, y

"""
Label-flipping attacks from Section 5.1 of WW-FL paper.
All attacks operate at data level (not model level).
"""
import random
import numpy as np
import torch


NUM_CLASSES = 10


def random_label_flip(label: int) -> int:
    """RLF [XXE12]: assign random class label."""
    return random.randint(0, NUM_CLASSES - 1)


def targeted_label_flip(label: int, source: int = 0, target: int = 1) -> int:
    """TLF [TTGL20]: flip source class → target class."""
    return target if label == source else label


def static_label_flip(label: int) -> int:
    """SLF [FCJG20]: fixed permutation new_label = num_classes - old_label - 1."""
    return NUM_CLASSES - label - 1


class DynamicLabelFlip:
    """
    DLF [SHKR22]: use a surrogate model trained on malicious clients' data
    to assign the least-probable class as the new label.
    Paper surrogate params: 50 epochs, bs=128, lr=0.05, momentum=0.9, wd=5e-4
    """
    def __init__(self, model_class, device="cpu"):
        self.model_class = model_class
        self.device = device
        self.surrogate = None

    def fit(self, dataloader):
        import torch.optim as optim
        import torch.nn as nn
        model = self.model_class().to(self.device)
        opt   = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        crit  = nn.CrossEntropyLoss()
        model.train()
        for epoch in range(50):
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                crit(model(x), y).backward()
                opt.step()
        self.surrogate = model

    def __call__(self, x_tensor: torch.Tensor) -> int:
        """Given a single sample tensor, return least-probable class."""
        assert self.surrogate is not None, "Call .fit() first"
        self.surrogate.eval()
        with torch.no_grad():
            logits = self.surrogate(x_tensor.unsqueeze(0).to(self.device))
            return logits.argmin(dim=1).item()


def get_poison_fn(attack: str, **kwargs):
    """
    Returns a label-transform callable (int -> int) for non-DLF attacks.
    For DLF, use DynamicLabelFlip class directly.
    """
    if attack == "none":
        return None
    elif attack == "rlf":
        return random_label_flip
    elif attack == "tlf":
        src = kwargs.get("source", 0)
        tgt = kwargs.get("target", 1)
        return lambda lbl: targeted_label_flip(lbl, src, tgt)
    elif attack == "slf":
        return static_label_flip
    else:
        raise ValueError(f"Unknown attack: {attack}. Use rlf/tlf/slf or DynamicLabelFlip for dlf.")

import pytest
import torch


@pytest.fixture(scope="module")
def num_items() -> int:
    """The number of candidate items to rank"""
    return 100000


@pytest.fixture(scope="module")
def cutoffs() -> torch.Tensor:
    """Cutoffs for the number of positions to include when computing metrics"""
    return torch.tensor([10, 20, 50, 100])


@pytest.fixture(scope="module")
def labels(num_items: int) -> torch.Tensor:
    """A large tensor of binary relevance labels"""
    probabilities = torch.empty(num_items).fill_(0.5)
    labels = torch.bernoulli(probabilities)
    return labels


@pytest.fixture(scope="module")
def scores(num_items: int) -> torch.Tensor:
    """A large tensor of simulated relevance scores"""
    return torch.empty(num_items).uniform_(0, 1)

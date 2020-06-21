import pytest
import torch

import hypothesis
from hypothesis import strategies as strat
from hypothesis import assume, given


@pytest.fixture(scope="session")
def num_items() -> int:
    """The number of candidate items to rank"""
    return 10000


@pytest.fixture(scope="session")
def batch_size() -> int:
    """The number of ranked lists to process"""
    return 10


@pytest.fixture(scope="module")
def cutoffs() -> torch.Tensor:
    """Cutoffs for the number of positions to include when computing metrics"""
    return torch.tensor([10, 20, 50, 100])


@pytest.fixture(scope="module")
def labels(batch_size: int, num_items: int) -> torch.Tensor:
    """A large tensor of binary relevance labels"""
    probabilities = torch.empty(batch_size, num_items).fill_(0.5)
    labels = torch.bernoulli(probabilities)
    return labels


@pytest.fixture(scope="module")
def scores(batch_size: int, num_items: int) -> torch.Tensor:
    """A large tensor of simulated relevance scores"""
    return torch.empty(batch_size, num_items).uniform_(0, 1)


@strat.composite
def scores_at_ks(draw, label_strat=strat.floats):
    """Test case generation strategy for cutoffs and pairs of true/pred score lists"""
    batch_size = draw(strat.integers(min_value=1, max_value=10))
    length = draw(strat.integers(min_value=2, max_value=10))

    cutoffs_list = strat.lists(
        strat.integers(min_value=length, max_value=length), min_size=1, max_size=10
    )

    scores_list = strat.lists(
        strat.lists(
            strat.floats(min_value=0, max_value=1),
            min_size=length,
            max_size=length,
            unique=True,
        ),
        min_size=batch_size,
        max_size=batch_size,
    )

    labels_list = strat.lists(
        strat.lists(
            label_strat(min_value=0, max_value=1), min_size=length, max_size=length,
        ),
        min_size=batch_size,
        max_size=batch_size,
    )

    return (draw(cutoffs_list), draw(scores_list), draw(labels_list))

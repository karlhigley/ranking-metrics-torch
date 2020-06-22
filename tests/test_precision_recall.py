import pytest
import torch

from ranking_metrics_torch.precision_recall import precision_at
from ranking_metrics_torch.precision_recall import recall_at


def test_precision_has_correct_shape(
    batch_size: int, cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, labels)
    assert len(p_at_ks.shape) == 2
    assert p_at_ks.shape == (batch_size, len(cutoffs))


def test_precision_when_nothing_is_relevant(
    batch_size: int, num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, torch.zeros(batch_size, num_items))
    assert (p_at_ks == torch.zeros(batch_size, len(cutoffs))).all()


def test_precision_when_everything_is_relevant(
    batch_size: int, num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, torch.ones(batch_size, num_items))
    assert (p_at_ks == torch.ones(batch_size, len(cutoffs))).all()


def test_precision_when_some_things_are_relevant(
    batch_size: int, cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, labels)
    assert (p_at_ks < torch.ones(batch_size, len(cutoffs))).any()
    assert (p_at_ks > torch.zeros(batch_size, len(cutoffs))).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_precision_works_on_gpu(
    batch_size: int, cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    gpu = torch.device("cuda")

    gpu_cutoffs = cutoffs.to(device=gpu)
    gpu_scores = scores.to(device=gpu)
    gpu_labels = labels.to(device=gpu)

    p_at_ks = precision_at(gpu_cutoffs, gpu_scores, gpu_labels)

    assert p_at_ks.device.type == gpu.type

    assert (p_at_ks < torch.ones(batch_size, len(cutoffs), device=gpu)).any()
    assert (p_at_ks > torch.zeros(batch_size, len(cutoffs), device=gpu)).any()


def test_recall_has_correct_shape(
    batch_size: int, cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, labels)
    assert len(r_at_ks.shape) == 2
    assert r_at_ks.shape == (batch_size, len(cutoffs))


def test_recall_when_nothing_is_relevant(
    batch_size: int, num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, torch.zeros(batch_size, num_items))
    assert (r_at_ks == torch.zeros(batch_size, len(cutoffs))).all()


def test_recall_when_everything_is_relevant(
    batch_size: int, num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, torch.ones(batch_size, num_items))
    expected = (
        (cutoffs.float() / num_items).repeat(batch_size, 1).to(dtype=torch.float64)
    )
    assert torch.allclose(r_at_ks, expected)


def test_recall_when_some_things_are_relevant(
    batch_size: int, cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, labels)
    assert (r_at_ks < torch.ones(batch_size, len(cutoffs))).all()
    assert (r_at_ks > torch.zeros(batch_size, len(cutoffs))).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_recall_works_on_gpu(
    batch_size: int, cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    gpu = torch.device("cuda")

    gpu_cutoffs = cutoffs.to(device=gpu)
    gpu_scores = scores.to(device=gpu)
    gpu_labels = labels.to(device=gpu)

    r_at_ks = recall_at(gpu_cutoffs, gpu_scores, gpu_labels)

    assert r_at_ks.device.type == gpu.type

    assert (r_at_ks < torch.ones(batch_size, len(cutoffs), device=gpu)).all()
    assert (r_at_ks > torch.zeros(batch_size, len(cutoffs), device=gpu)).all()

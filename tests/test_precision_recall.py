import pytest
import torch

from ranking_metrics_torch.precision_recall import precision_at
from ranking_metrics_torch.precision_recall import recall_at


def test_precision_has_correct_shape(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, labels)
    assert len(p_at_ks.shape) == 1
    assert p_at_ks.shape[0] == len(cutoffs)


def test_precision_when_nothing_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, torch.zeros(num_items))
    assert all(p_at_ks == torch.zeros(len(cutoffs)))


def test_precision_when_everything_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, torch.ones(num_items))
    assert all(p_at_ks == torch.ones(len(cutoffs)))


def test_precision_when_some_things_are_relevant(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    p_at_ks = precision_at(cutoffs, scores, labels)
    assert all(p_at_ks < torch.ones(len(cutoffs)))
    assert all(p_at_ks > torch.zeros(len(cutoffs)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_precision_works_on_gpu(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    gpu = torch.device("cuda")

    gpu_cutoffs = cutoffs.to(device=gpu)
    gpu_scores = scores.to(device=gpu)
    gpu_labels = labels.to(device=gpu)

    p_at_ks = precision_at(gpu_cutoffs, gpu_scores, gpu_labels)

    assert p_at_ks.device.type == gpu.type

    assert all(p_at_ks < torch.ones(len(cutoffs), device=gpu))
    assert all(p_at_ks > torch.zeros(len(cutoffs), device=gpu))


def test_recall_has_correct_shape(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, labels)
    assert len(r_at_ks.shape) == 1
    assert r_at_ks.shape[0] == len(cutoffs)


def test_recall_when_nothing_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, torch.zeros(num_items))
    assert all(r_at_ks == torch.zeros(len(cutoffs)))


def test_recall_when_everything_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, torch.ones(num_items))
    assert all(r_at_ks == cutoffs.float() / num_items)


def test_recall_when_some_things_are_relevant(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    r_at_ks = recall_at(cutoffs, scores, labels)
    assert all(r_at_ks < torch.ones(len(cutoffs)))
    assert all(r_at_ks > torch.zeros(len(cutoffs)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_recall_works_on_gpu(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    gpu = torch.device("cuda")

    gpu_cutoffs = cutoffs.to(device=gpu)
    gpu_scores = scores.to(device=gpu)
    gpu_labels = labels.to(device=gpu)

    r_at_ks = recall_at(gpu_cutoffs, gpu_scores, gpu_labels)

    assert r_at_ks.device.type == gpu.type

    assert all(r_at_ks < torch.ones(len(cutoffs), device=gpu))
    assert all(r_at_ks > torch.zeros(len(cutoffs), device=gpu))

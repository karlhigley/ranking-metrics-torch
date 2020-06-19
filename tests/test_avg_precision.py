import pytest
import torch

import hypothesis
from hypothesis import strategies as strat
from hypothesis import assume, given

import sklearn
from sklearn.metrics import average_precision_score

from ranking_metrics_torch.avg_precision import avg_precision_at
from tests.conftest import scores_at_ks


def test_avg_precision_has_correct_shape(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    ap_at_ks = avg_precision_at(cutoffs, scores, labels)
    assert len(ap_at_ks.shape) == 1
    assert ap_at_ks.shape[0] == len(cutoffs)


def test_avg_precision_when_nothing_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    ap_at_ks = avg_precision_at(cutoffs, scores, torch.zeros(num_items))
    assert all(ap_at_ks == torch.zeros(len(cutoffs)))


def test_avg_precision_when_everything_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    ap_at_ks = avg_precision_at(cutoffs, scores, torch.ones(num_items))
    assert all(ap_at_ks == torch.ones(len(cutoffs)))


def test_avg_precision_when_some_things_are_relevant(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    ap_at_ks = avg_precision_at(cutoffs, scores, labels)
    assert all(ap_at_ks < torch.ones(len(cutoffs)))
    assert all(ap_at_ks > torch.zeros(len(cutoffs)))


@given(scores_at_ks(label_strat=strat.integers))
def test_avg_precision_matches_sklearn(scores_at_ks):
    ks, pred_scores, true_scores = scores_at_ks

    assume(sum(pred_scores) > 0)
    assume(sum(true_scores) > 0)
    assume(len(set(pred_scores)) == len(pred_scores))

    avg_precisions_at_k = avg_precision_at(
        ks,
        torch.tensor(pred_scores, dtype=torch.float64),
        torch.tensor(true_scores, dtype=torch.float64),
    )

    sklearn_avg_precisions_at_k = [
        average_precision_score(true_scores[:k], pred_scores[:k]) for k in ks
    ]

    assert avg_precisions_at_k.tolist() == pytest.approx(sklearn_avg_precisions_at_k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_avg_precision_works_on_gpu(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    gpu = torch.device("cuda")

    gpu_cutoffs = cutoffs.to(device=gpu)
    gpu_scores = scores.to(device=gpu)
    gpu_labels = labels.to(device=gpu)

    ap_at_ks = avg_precision_at(gpu_cutoffs, gpu_scores, gpu_labels)

    assert ap_at_ks.device.type == gpu.type

    assert all(ap_at_ks < torch.ones(len(cutoffs), device=gpu))
    assert all(ap_at_ks > torch.zeros(len(cutoffs), device=gpu))

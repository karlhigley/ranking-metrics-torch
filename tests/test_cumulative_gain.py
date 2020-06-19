import pytest
import torch

import hypothesis
from hypothesis import strategies as strat
from hypothesis import assume, given

import sklearn
from sklearn.metrics import dcg_score
from sklearn.metrics import ndcg_score

from ranking_metrics_torch.cumulative_gain import dcg_at
from ranking_metrics_torch.cumulative_gain import ndcg_at
from tests.conftest import scores_at_ks


def test_dcg_has_correct_shape(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    dcg_at_ks = dcg_at(cutoffs, scores, labels)
    assert len(dcg_at_ks.shape) == 1
    assert dcg_at_ks.shape[0] == len(cutoffs)


def test_dcg_when_nothing_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    dcg_at_ks = dcg_at(cutoffs, scores, torch.zeros(num_items))
    assert all(dcg_at_ks == torch.zeros(len(cutoffs)))


@given(scores_at_ks())
def test_dcg_matches_sklearn(scores_at_ks):
    ks, pred_scores, true_scores = scores_at_ks

    assume(sum(pred_scores) > 0)
    assume(len(set(pred_scores)) == len(pred_scores))

    dcgs_at_k = dcg_at(
        ks,
        torch.tensor(pred_scores, dtype=torch.float64),
        torch.tensor(true_scores, dtype=torch.float64),
    )

    sklearn_dcgs_at_k = [
        dcg_score([true_scores], [pred_scores], k=k, ignore_ties=True) for k in ks
    ]

    assert dcgs_at_k.tolist() == pytest.approx(sklearn_dcgs_at_k)


def test_ndcg_has_correct_shape(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    ndcg_at_ks = ndcg_at(cutoffs, scores, labels)
    assert len(ndcg_at_ks.shape) == 1
    assert ndcg_at_ks.shape[0] == len(cutoffs)


def test_ndcg_when_nothing_is_relevant(
    num_items: int, cutoffs: torch.Tensor, scores: torch.Tensor
) -> None:

    ndcg_at_ks = ndcg_at(cutoffs, scores, torch.zeros(num_items))
    assert all(ndcg_at_ks == torch.zeros(len(cutoffs)))


@given(scores_at_ks())
def test_ndcg_matches_sklearn(scores_at_ks):
    ks, pred_scores, true_scores = scores_at_ks

    assume(sum(pred_scores) > 0)
    assume(len(set(pred_scores)) == len(pred_scores))

    ndcgs_at_k = ndcg_at(
        ks,
        torch.tensor(pred_scores, dtype=torch.float64),
        torch.tensor(true_scores, dtype=torch.float64),
    )

    sklearn_ndcgs_at_k = [
        ndcg_score([true_scores], [pred_scores], k=k, ignore_ties=True) for k in ks
    ]

    assert ndcgs_at_k.tolist() == pytest.approx(sklearn_ndcgs_at_k)

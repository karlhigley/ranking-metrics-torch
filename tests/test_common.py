import pytest
import torch

from ranking_metrics_torch.common import _check_inputs


def test_check_inputs_rejects_2d_cutoffs(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    cutoffs = torch.stack([cutoffs, cutoffs])

    with pytest.raises(ValueError, match=r"1-dimensional"):
        _check_inputs(cutoffs, scores, labels)


def test_check_inputs_rejects_1d_scores(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    scores = scores[0]

    with pytest.raises(ValueError, match=r"2-dimensional"):
        _check_inputs(cutoffs, scores, labels)


def test_check_inputs_rejects_1d_labels(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    labels = labels[0]

    with pytest.raises(ValueError, match=r"2-dimensional"):
        _check_inputs(cutoffs, scores, labels)


def test_check_inputs_rejects_mismatched_scores_and_labels(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    labels = torch.cat([labels, labels])

    with pytest.raises(ValueError, match=r"same shape"):
        _check_inputs(cutoffs, scores, labels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
def test_check_inputs_moves_args_to_scores_device(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    scores = scores.cuda()

    ks, scores, labels = _check_inputs(cutoffs, scores, labels)

    assert ks.device == scores.device
    assert labels.device == scores.device


def test_check_inputs_converts_to_float(
    cutoffs: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor
) -> None:

    scores = scores.to(dtype=torch.int32)
    labels = labels.to(dtype=torch.int32)

    ks, scores, labels = _check_inputs(cutoffs, scores, labels)

    assert scores.dtype == torch.float
    assert labels.dtype == torch.float
